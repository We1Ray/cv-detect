"""REST API server for CV defect detection.

Provides HTTP endpoints for remote inspection, model management,
and SPC data retrieval. Designed for MES/ERP integration.

Usage:
    python -m api.server                    # default port 8000
    python -m api.server --port 9000        # custom port
    python -m api.server --host 0.0.0.0     # listen on all interfaces
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path so that ``dl_anomaly`` and ``variation_model``
# packages can be imported regardless of the working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Fix #12: Add VM root to sys.path once at module level so that bare
# imports inside the variation_model package (``from config import ...``,
# ``from core.inspector import ...``) resolve correctly.
_VM_ROOT = str(Path(_PROJECT_ROOT) / "variation_model")
if _VM_ROOT not in sys.path:
    sys.path.insert(0, _VM_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix #8: single version constant
# ---------------------------------------------------------------------------
_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Fix #3: upload size limit
# ---------------------------------------------------------------------------
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Fix #4: batch file count limit
# ---------------------------------------------------------------------------
_MAX_BATCH_FILES = 100

# ---------------------------------------------------------------------------
# Fix #2: allowed model directories for path traversal protection
# ---------------------------------------------------------------------------
_ALLOWED_MODEL_DIRS = [
    Path(_PROJECT_ROOT, "models").resolve(),
    Path(_PROJECT_ROOT, "dl_anomaly", "checkpoints").resolve(),
    Path(_PROJECT_ROOT, "variation_model", "models").resolve(),
]

# ---------------------------------------------------------------------------
# Module-level model state (guarded by a lock for thread safety)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_dl_pipeline: Any = None  # dl_anomaly.pipeline.inference.InferencePipeline
_vm_model: Any = None  # variation_model.core.variation_model.VariationModel
_vm_config: Any = None  # variation_model.config.Config
_vm_pipeline: Any = None  # variation_model.pipeline.inference.InferencePipeline
_results_db: Any = None  # shared.core.results_db.ResultsDatabase

# Fix #11: will be set by startup event
_start_time: Optional[datetime] = None

# Default database path
_DB_PATH = str(Path(_PROJECT_ROOT) / "results" / "inspection_results.db")


def _get_results_db():
    """Lazily initialise and return the ResultsDatabase singleton."""
    global _results_db
    with _lock:
        if _results_db is None:
            from shared.core.results_db import ResultsDatabase
            _results_db = ResultsDatabase(db_path=_DB_PATH)
        return _results_db


def _is_path_allowed(p: Path) -> bool:
    """Return True if *p* is inside one of the allowed model directories."""
    resolved = p.resolve()
    return any(
        resolved == allowed or allowed in resolved.parents
        for allowed in _ALLOWED_MODEL_DIRS
    )


def _get_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for the API server. "
            "Install with: pip install fastapi uvicorn python-multipart"
        )

    @asynccontextmanager
    async def _lifespan(application):
        global _start_time
        _start_time = datetime.now()
        yield

    app = FastAPI(
        title="CV Defect Detection API",
        description="Industrial inspection REST API for MES/ERP integration",
        version=_VERSION,
        lifespan=_lifespan,
    )

    # -------------------------------------------------------------------
    # Fix #7: CORS middleware
    # -------------------------------------------------------------------
    origins = [o.strip() for o in os.environ.get("CV_DETECT_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -------------------------------------------------------------------
    # Fix #1: API Key Authentication
    # -------------------------------------------------------------------
    _api_key = os.environ.get("CV_DETECT_API_KEY") or None
    _bearer_scheme = HTTPBearer(auto_error=False)

    async def _verify_api_key(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ):
        """Check the Authorization header when CV_DETECT_API_KEY is set."""
        if _api_key is None:
            # Auth disabled in dev mode
            return
        import hmac
        if credentials is None or not hmac.compare_digest(credentials.credentials, _api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    # -------------------------------------------------------------------
    # Pydantic response models
    # -------------------------------------------------------------------

    class InspectionResult(BaseModel):
        is_defective: bool
        score: float
        defect_count: int
        defect_regions: List[Dict[str, Any]]
        model_type: str
        timestamp: str
        processing_time_ms: float

    class ModelInfo(BaseModel):
        model_type: str
        is_loaded: bool
        details: Dict[str, Any]

    class HealthResponse(BaseModel):
        status: str
        version: str
        models_loaded: Dict[str, bool]
        uptime_seconds: float

    class SPCResponse(BaseModel):
        mean: float
        std: float
        ucl: float
        lcl: float
        cpk: Optional[float] = None
        yield_rate: float
        total_inspections: int
        recent_results: List[Dict[str, Any]]

    class BatchInspectionResponse(BaseModel):
        total: int
        results: List[InspectionResult]
        total_processing_time_ms: float

    class LoadModelResponse(BaseModel):
        status: str
        message: str

    # -------------------------------------------------------------------
    # Helper: save UploadFile to a temporary path
    # Fix #3: chunked read with size limit and content-type validation
    # -------------------------------------------------------------------
    async def _save_upload_to_temp(file: UploadFile) -> Path:
        """Read an uploaded file into a named temporary file and return its path."""
        # Validate content type
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type '{content_type}'. Only image/* is accepted.",
            )

        suffix = Path(file.filename or "image.png").suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        total_size = 0
        try:
            while True:
                chunk = await file.read(1024 * 256)  # 256 KB chunks
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > _MAX_UPLOAD_BYTES:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size of {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                tmp.write(chunk)
            tmp.flush()
        finally:
            tmp.close()

        if total_size == 0:
            Path(tmp.name).unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        return Path(tmp.name)

    # -------------------------------------------------------------------
    # Helper: run DL inspection on a single image path
    # -------------------------------------------------------------------
    def _run_dl_inspection(image_path: Path) -> InspectionResult:
        """Execute the DL (autoencoder) pipeline on *image_path*."""
        with _lock:
            pipeline = _dl_pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="DL model is not loaded. POST to /models/dl/load first.",
            )

        t0 = time.time()
        result = pipeline.inspect_single(image_path)
        elapsed_ms = (time.time() - t0) * 1000.0

        # Serialise defect_regions -- convert tuples to lists for JSON
        regions = []
        for r in result.defect_regions:
            region_copy: Dict[str, Any] = {}
            for k, v in r.items():
                if isinstance(v, (tuple, list)):
                    region_copy[k] = list(v)
                else:
                    region_copy[k] = v
            regions.append(region_copy)

        return InspectionResult(
            is_defective=bool(result.is_defective),
            score=float(result.anomaly_score),
            defect_count=len(result.defect_regions),
            defect_regions=regions,
            model_type="autoencoder",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
        )

    # -------------------------------------------------------------------
    # Helper: run VM inspection on a single image path
    # -------------------------------------------------------------------
    def _run_vm_inspection(image_path: Path) -> InspectionResult:
        """Execute the Variation Model pipeline on *image_path*."""
        with _lock:
            pipeline = _vm_pipeline
        if pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="VM model is not loaded. POST to /models/vm/load first.",
            )

        t0 = time.time()
        result, _ = pipeline.inspect_single(image_path)
        elapsed_ms = (time.time() - t0) * 1000.0

        # Serialise defect_regions
        regions = []
        for r in result.defect_regions:
            region_copy: Dict[str, Any] = {}
            for k, v in r.items():
                if isinstance(v, (tuple, list)):
                    region_copy[k] = list(v)
                elif hasattr(v, "item"):
                    # Convert numpy scalar types (np.int64, np.float32, etc.)
                    region_copy[k] = v.item()
                else:
                    region_copy[k] = v
            regions.append(region_copy)

        return InspectionResult(
            is_defective=bool(result.is_defective),
            score=float(result.score),
            defect_count=result.num_defects,
            defect_regions=regions,
            model_type="variation_model",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
        )

    # ===================================================================
    # ENDPOINTS
    # ===================================================================

    # --- Health --------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        # Fix #6: thread-safe reads of pipeline references
        with _lock:
            dl_loaded = _dl_pipeline is not None
            vm_loaded = _vm_pipeline is not None
        uptime = (datetime.now() - (_start_time or datetime.now())).total_seconds()
        return HealthResponse(
            status="ok",
            version=_VERSION,
            models_loaded={
                "dl_autoencoder": dl_loaded,
                "variation_model": vm_loaded,
            },
            uptime_seconds=round(uptime, 2),
        )

    # --- Inspection: DL ------------------------------------------------

    @app.post("/inspect/dl", response_model=InspectionResult, dependencies=[Depends(_verify_api_key)])
    async def inspect_dl(file: UploadFile = File(...)):
        """Run DL (autoencoder) inspection on an uploaded image."""
        tmp_path = await _save_upload_to_temp(file)
        try:
            # Fix #10: run sync inference in thread pool
            result = await asyncio.get_running_loop().run_in_executor(
                None, _run_dl_inspection, tmp_path,
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("DL inspection failed")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Inspection: VM ------------------------------------------------

    @app.post("/inspect/vm", response_model=InspectionResult, dependencies=[Depends(_verify_api_key)])
    async def inspect_vm(file: UploadFile = File(...)):
        """Run Variation Model inspection on an uploaded image."""
        tmp_path = await _save_upload_to_temp(file)
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, _run_vm_inspection, tmp_path,
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VM inspection failed")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Inspection: Auto ----------------------------------------------

    @app.post("/inspect/auto", response_model=InspectionResult, dependencies=[Depends(_verify_api_key)])
    async def inspect_auto(file: UploadFile = File(...)):
        """Auto-select the best available model for inspection.

        Preference order: DL autoencoder > Variation Model.
        """
        # Fix #6: thread-safe reads
        with _lock:
            dl_loaded = _dl_pipeline is not None
            vm_loaded = _vm_pipeline is not None

        if not dl_loaded and not vm_loaded:
            raise HTTPException(
                status_code=503,
                detail="No model is loaded. Load a model first via /models/dl/load or /models/vm/load.",
            )

        tmp_path = await _save_upload_to_temp(file)
        try:
            if dl_loaded:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, _run_dl_inspection, tmp_path,
                )
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, _run_vm_inspection, tmp_path,
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Auto inspection failed")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Model Management ----------------------------------------------

    @app.get("/models", response_model=List[ModelInfo], dependencies=[Depends(_verify_api_key)])
    async def list_models():
        """List all model slots and their current status."""
        # Fix #6: thread-safe reads
        with _lock:
            dl_ref = _dl_pipeline
            vm_model_ref = _vm_model
            vm_pipe_ref = _vm_pipeline

        models: List[ModelInfo] = []

        # DL autoencoder
        dl_details: Dict[str, Any] = {}
        if dl_ref is not None:
            try:
                dl_details = {
                    "device": str(dl_ref.device),
                    "image_size": dl_ref.config.image_size,
                    "grayscale": dl_ref.config.grayscale,
                    "threshold": dl_ref.scorer.threshold,
                }
            except Exception:
                dl_details = {"note": "loaded but details unavailable"}
        models.append(ModelInfo(
            model_type="dl_autoencoder",
            is_loaded=dl_ref is not None,
            details=dl_details,
        ))

        # Variation Model
        vm_details: Dict[str, Any] = {}
        if vm_model_ref is not None:
            try:
                vm_details = {
                    "training_count": vm_model_ref.count,
                    "is_trained": vm_model_ref.is_trained,
                    "abs_threshold": vm_model_ref._abs_threshold,
                    "var_threshold": vm_model_ref._var_threshold,
                }
            except Exception:
                vm_details = {"note": "loaded but details unavailable"}
        models.append(ModelInfo(
            model_type="variation_model",
            is_loaded=vm_pipe_ref is not None,
            details=vm_details,
        ))

        return models

    @app.post("/models/dl/load", response_model=LoadModelResponse, dependencies=[Depends(_verify_api_key)])
    async def load_dl_model(
        checkpoint_path: str = Query(..., description="Path to a .pt checkpoint on the server"),
    ):
        """Load a DL autoencoder checkpoint from a path on the server."""
        global _dl_pipeline
        cp = Path(checkpoint_path).resolve()

        # Fix #2: path traversal protection
        if not _is_path_allowed(cp):
            raise HTTPException(
                status_code=403,
                detail="Model path is outside the allowed directories.",
            )
        if not cp.exists():
            raise HTTPException(
                status_code=404,
                detail="Checkpoint not found.",
            )

        try:
            from dl_anomaly.pipeline.inference import InferencePipeline as DLInferencePipeline
            pipeline = DLInferencePipeline(cp)
            with _lock:
                _dl_pipeline = pipeline
            logger.info("DL model loaded from %s", cp)
            return LoadModelResponse(
                status="ok",
                message="DL autoencoder loaded successfully.",
            )
        except Exception as exc:
            logger.exception("Failed to load DL model")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )

    @app.post("/models/vm/load", response_model=LoadModelResponse, dependencies=[Depends(_verify_api_key)])
    async def load_vm_model(
        model_path: str = Query(..., description="Path to a .npz model file on the server"),
    ):
        """Load a Variation Model (.npz) from a path on the server."""
        global _vm_model, _vm_config, _vm_pipeline
        mp = Path(model_path).resolve()

        # Fix #2: path traversal protection
        if not _is_path_allowed(mp):
            raise HTTPException(
                status_code=403,
                detail="Model path is outside the allowed directories.",
            )
        if not mp.exists() and not mp.with_suffix(".npz").exists():
            raise HTTPException(
                status_code=404,
                detail="Model file not found.",
            )

        try:
            # Fix #12: sys.path already set at module level
            from core.variation_model import VariationModel  # type: ignore[import-untyped]
            from config import Config as VMConfig  # type: ignore[import-untyped]
            from pipeline.inference import InferencePipeline as VMInferencePipeline  # type: ignore[import-untyped]

            model = VariationModel.load(mp)
            config = VMConfig()
            pipeline = VMInferencePipeline(model, config)

            with _lock:
                _vm_model = model
                _vm_config = config
                _vm_pipeline = pipeline

            logger.info("VM model loaded from %s (training_count=%d)", mp, model.count)
            return LoadModelResponse(
                status="ok",
                message=f"Variation Model loaded ({model.count} training images).",
            )
        except Exception as exc:
            logger.exception("Failed to load VM model")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )

    # --- SPC -----------------------------------------------------------

    @app.get("/spc/metrics", response_model=SPCResponse, dependencies=[Depends(_verify_api_key)])
    async def spc_metrics():
        """Get current SPC metrics from the inspection results database."""
        try:
            db = _get_results_db()
            spc = db.compute_spc_metrics(field="anomaly_score")

            # Fix #9: use SQL COUNT via db.get_summary() instead of loading all records
            summary = db.get_summary()
            total = summary.get("total", 0)
            pass_count = summary.get("pass", 0)
            yield_rate = pass_count / total if total > 0 else 0.0

            # Recent results for the response
            recent_records = db.query_records(limit=20)
            recent = [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "model_type": r.model_type,
                    "anomaly_score": r.anomaly_score,
                    "is_defective": r.is_defective,
                    "defect_count": r.defect_count,
                }
                for r in recent_records
            ]

            return SPCResponse(
                mean=round(spc.mean, 6),
                std=round(spc.std, 6),
                ucl=round(spc.ucl, 6),
                lcl=round(spc.lcl, 6),
                cpk=round(spc.cpk, 4) if spc.cpk is not None else None,
                yield_rate=round(yield_rate, 4),
                total_inspections=total,
                recent_results=recent,
            )
        except Exception as exc:
            logger.exception("Failed to compute SPC metrics")
            raise HTTPException(
                status_code=500,
                detail="Internal processing error. Check server logs for details.",
            )

    # --- Batch Inspection ----------------------------------------------

    @app.post("/inspect/batch", response_model=BatchInspectionResponse, dependencies=[Depends(_verify_api_key)])
    async def inspect_batch(files: List[UploadFile] = File(...)):
        """Batch-inspect multiple uploaded images.

        Uses the best available model (DL preferred over VM).
        """
        # Fix #6: thread-safe reads
        with _lock:
            dl_loaded = _dl_pipeline is not None
            vm_loaded = _vm_pipeline is not None

        if not dl_loaded and not vm_loaded:
            raise HTTPException(
                status_code=503,
                detail="No model is loaded. Load a model first.",
            )

        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Fix #4: batch file count limit
        if len(files) > _MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum is {_MAX_BATCH_FILES}.",
            )

        use_dl = dl_loaded
        results: List[InspectionResult] = []
        t0_total = time.time()

        for file in files:
            tmp_path = await _save_upload_to_temp(file)
            try:
                if use_dl:
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, _run_dl_inspection, tmp_path,
                    )
                else:
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, _run_vm_inspection, tmp_path,
                    )
                results.append(result)
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("Batch inspection failed for %s: %s", file.filename, exc)
                # Append a failure placeholder so the caller knows which file failed
                results.append(InspectionResult(
                    is_defective=False,
                    score=0.0,
                    defect_count=0,
                    defect_regions=[],
                    model_type="error",
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0.0,
                ))
            finally:
                tmp_path.unlink(missing_ok=True)

        total_ms = (time.time() - t0_total) * 1000.0

        return BatchInspectionResponse(
            total=len(results),
            results=results,
            total_processing_time_ms=round(total_ms, 2),
        )

    return app


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main():
    """Run the API server via uvicorn."""
    parser = argparse.ArgumentParser(description="CV Defect Detection API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the inspection results SQLite database",
    )
    args = parser.parse_args()

    # Override default DB path if provided
    if args.db_path is not None:
        global _DB_PATH
        _DB_PATH = args.db_path

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "api.server:_get_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
