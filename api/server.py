"""REST API server for CV defect detection.

Provides HTTP endpoints for remote inspection, model management,
and SPC data retrieval. Designed for MES/ERP integration.

Usage:
    python -m api.server                    # default port 8000
    python -m api.server --port 9000        # custom port
    python -m api.server --host 0.0.0.0     # listen on all interfaces
"""

import argparse
import logging
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path so that ``dl_anomaly`` and ``variation_model``
# packages can be imported regardless of the working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model state (guarded by a lock for thread safety)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_dl_pipeline: Any = None  # dl_anomaly.pipeline.inference.InferencePipeline
_vm_model: Any = None  # variation_model.core.variation_model.VariationModel
_vm_config: Any = None  # variation_model.config.Config
_vm_pipeline: Any = None  # variation_model.pipeline.inference.InferencePipeline
_results_db: Any = None  # shared.core.results_db.ResultsDatabase

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


def _get_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, File, HTTPException, Query, UploadFile
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for the API server. "
            "Install with: pip install fastapi uvicorn python-multipart"
        )

    app = FastAPI(
        title="CV Defect Detection API",
        description="Industrial inspection REST API for MES/ERP integration",
        version="2.0.0",
    )

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
    # Startup timestamp
    # -------------------------------------------------------------------
    _start_time = datetime.now()

    # -------------------------------------------------------------------
    # Helper: save UploadFile to a temporary path
    # -------------------------------------------------------------------
    async def _save_upload_to_temp(file: UploadFile) -> Path:
        """Read an uploaded file into a named temporary file and return its path."""
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        suffix = Path(file.filename or "image.png").suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    # -------------------------------------------------------------------
    # Helper: run DL inspection on a single image path
    # -------------------------------------------------------------------
    def _run_dl_inspection(image_path: Path) -> InspectionResult:
        """Execute the DL (autoencoder) pipeline on *image_path*."""
        global _dl_pipeline
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
        global _vm_pipeline
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
        uptime = (datetime.now() - _start_time).total_seconds()
        return HealthResponse(
            status="ok",
            version="2.0.0",
            models_loaded={
                "dl_autoencoder": _dl_pipeline is not None,
                "variation_model": _vm_pipeline is not None,
            },
            uptime_seconds=round(uptime, 2),
        )

    # --- Inspection: DL ------------------------------------------------

    @app.post("/inspect/dl", response_model=InspectionResult)
    async def inspect_dl(file: UploadFile = File(...)):
        """Run DL (autoencoder) inspection on an uploaded image."""
        tmp_path = await _save_upload_to_temp(file)
        try:
            return _run_dl_inspection(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("DL inspection failed")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Inspection: VM ------------------------------------------------

    @app.post("/inspect/vm", response_model=InspectionResult)
    async def inspect_vm(file: UploadFile = File(...)):
        """Run Variation Model inspection on an uploaded image."""
        tmp_path = await _save_upload_to_temp(file)
        try:
            return _run_vm_inspection(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VM inspection failed")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Inspection: Auto ----------------------------------------------

    @app.post("/inspect/auto", response_model=InspectionResult)
    async def inspect_auto(file: UploadFile = File(...)):
        """Auto-select the best available model for inspection.

        Preference order: DL autoencoder > Variation Model.
        """
        if _dl_pipeline is None and _vm_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="No model is loaded. Load a model first via /models/dl/load or /models/vm/load.",
            )

        tmp_path = await _save_upload_to_temp(file)
        try:
            if _dl_pipeline is not None:
                return _run_dl_inspection(tmp_path)
            else:
                return _run_vm_inspection(tmp_path)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Auto inspection failed")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

    # --- Model Management ----------------------------------------------

    @app.get("/models", response_model=List[ModelInfo])
    async def list_models():
        """List all model slots and their current status."""
        models: List[ModelInfo] = []

        # DL autoencoder
        dl_details: Dict[str, Any] = {}
        if _dl_pipeline is not None:
            try:
                dl_details = {
                    "device": str(_dl_pipeline.device),
                    "image_size": _dl_pipeline.config.image_size,
                    "grayscale": _dl_pipeline.config.grayscale,
                    "threshold": _dl_pipeline.scorer.threshold,
                }
            except Exception:
                dl_details = {"note": "loaded but details unavailable"}
        models.append(ModelInfo(
            model_type="dl_autoencoder",
            is_loaded=_dl_pipeline is not None,
            details=dl_details,
        ))

        # Variation Model
        vm_details: Dict[str, Any] = {}
        if _vm_model is not None:
            try:
                vm_details = {
                    "training_count": _vm_model.count,
                    "is_trained": _vm_model.is_trained,
                    "abs_threshold": _vm_model._abs_threshold,
                    "var_threshold": _vm_model._var_threshold,
                }
            except Exception:
                vm_details = {"note": "loaded but details unavailable"}
        models.append(ModelInfo(
            model_type="variation_model",
            is_loaded=_vm_pipeline is not None,
            details=vm_details,
        ))

        return models

    @app.post("/models/dl/load", response_model=LoadModelResponse)
    async def load_dl_model(
        checkpoint_path: str = Query(..., description="Absolute path to a .pt checkpoint on the server"),
    ):
        """Load a DL autoencoder checkpoint from a path on the server."""
        global _dl_pipeline
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {checkpoint_path}",
            )

        try:
            from dl_anomaly.pipeline.inference import InferencePipeline as DLInferencePipeline
            pipeline = DLInferencePipeline(cp)
            with _lock:
                _dl_pipeline = pipeline
            logger.info("DL model loaded from %s", cp)
            return LoadModelResponse(
                status="ok",
                message=f"DL autoencoder loaded from {cp}",
            )
        except Exception as exc:
            logger.exception("Failed to load DL model")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/models/vm/load", response_model=LoadModelResponse)
    async def load_vm_model(
        model_path: str = Query(..., description="Absolute path to a .npz model file on the server"),
    ):
        """Load a Variation Model (.npz) from a path on the server."""
        global _vm_model, _vm_config, _vm_pipeline
        mp = Path(model_path)
        if not mp.exists() and not mp.with_suffix(".npz").exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {model_path}",
            )

        try:
            # The VM package uses bare imports (``from config import Config``,
            # ``from core.inspector import ...``), so its own root directory
            # must be on sys.path for those to resolve correctly.
            vm_root = str(Path(_PROJECT_ROOT) / "variation_model")
            if vm_root not in sys.path:
                sys.path.insert(0, vm_root)

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
                message=f"Variation Model loaded from {mp} ({model.count} training images)",
            )
        except Exception as exc:
            logger.exception("Failed to load VM model")
            raise HTTPException(status_code=500, detail=str(exc))

    # --- SPC -----------------------------------------------------------

    @app.get("/spc/metrics", response_model=SPCResponse)
    async def spc_metrics():
        """Get current SPC metrics from the inspection results database."""
        try:
            db = _get_results_db()
            spc = db.compute_spc_metrics(field="anomaly_score")

            # Compute yield rate
            records = db.query_records(limit=100_000)
            total = len(records)
            if total > 0:
                pass_count = sum(1 for r in records if not r.is_defective)
                yield_rate = pass_count / total
            else:
                yield_rate = 0.0

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
            raise HTTPException(status_code=500, detail=str(exc))

    # --- Batch Inspection ----------------------------------------------

    @app.post("/inspect/batch", response_model=BatchInspectionResponse)
    async def inspect_batch(files: List[UploadFile] = File(...)):
        """Batch-inspect multiple uploaded images.

        Uses the best available model (DL preferred over VM).
        """
        if _dl_pipeline is None and _vm_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="No model is loaded. Load a model first.",
            )

        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        use_dl = _dl_pipeline is not None
        results: List[InspectionResult] = []
        t0_total = time.time()

        for file in files:
            tmp_path = await _save_upload_to_temp(file)
            try:
                if use_dl:
                    result = _run_dl_inspection(tmp_path)
                else:
                    result = _run_vm_inspection(tmp_path)
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
