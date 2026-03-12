# /workflow - 專案自動化工作流程

**統一入口：開發流程、測試、審查、部署等所有自動化任務。**

---

## 使用方式

| 指令 | 說明 |
|------|------|
| `/workflow <需求>` | 完整審視模式 (Phase 1-7) |
| `/workflow fix <問題>` | 快速修復模式 (Phase 3-5.5, 7) |
| `/workflow test [scope]` | 測試執行 |
| `/workflow check` | 程式碼檢查 |
| `/workflow review [files]` | 程式碼審查 |
| `/workflow security [scope]` | 安全審計 |
| `/workflow docs [type]` | 文件生成 |
| `/workflow deploy [env]` | 服務部署 |
| `/workflow setup [layer]` | 開發環境設置 |
| `/workflow llm <需求>` | LLM 專項開發 |
| `/workflow --silent` | 靜默執行，僅輸出最終報告 |
| `/workflow --no-restart` | 跳過 Phase 7 |

---

## Agent 角色

| Agent | 負責 Layer | 主要職責 |
|-------|-----------|---------|
| `system-architect` | All | 架構審視、跨層協調、影響分析 |
| `rust-developer` | Backend | Rust API Gateway 開發測試 |
| `python-developer` | LLM | Python Service 開發測試 |
| `flutter-agent` | Frontend | Flutter UI/State/Widget |
| `data-processor` | Database | Schema/Migration/RLS |
| `security-engineer` | All | 安全審計、漏洞掃描 |
| `qa-engineer` | All | 測試策略、覆蓋率、驗收 |
| `llm-researcher` | LLM | Prompt 工程、Token 優化 |
| `system-admin` | Ops | 部署、監控、基礎設施 |

---

## 核心原則

### 跨層資料格式

| PostgreSQL | Rust | Python | Dart |
|------------|------|--------|------|
| `VARCHAR` | `String` | `str` | `String` |
| `UUID` | `Uuid` | `str` | `String` |
| `TIMESTAMPTZ` | `DateTime<Utc>` | `datetime` | `DateTime` |
| `BOOLEAN` | `bool` | `bool` | `bool` |
| `JSONB` | `serde_json::Value` | `dict` | `Map<String, dynamic>` |

**JSON 命名**: PostgreSQL/Rust/Python = `snake_case`, Flutter class = `camelCase`, Flutter JSON = `@JsonKey(name: 'snake_case')`

**Migration 命名**: `NN_table_name.sql` (禁用 fix/patch/update)

---

## Phase 依賴關係

```
Phase 1 (狀態檢視)
    ▼
Phase 2 (文件同步) ────────────────────┐
    ▼                                   │
Phase 3 (程式碼檢查) ─┐                 │
    ▼                  │ 問題清單         │
Phase 4 (測試執行) ──→ Phase 5 (修復) ←─┘
                          ▼
                    Phase 5.5 (Memory Bank)
                          ▼
                    Phase 6 (最終驗證)
                          ▼
                    Phase 7 (服務部署)
```

---

## Phase 1: 專案狀態檢視

**Lead: system-architect**

1. `git status` + `git log --oneline -10`
2. 讀取 `docs/ARCHITECTURE.md`, `DATABASE.md`, `FEATURES.md`, `SECURITY.md`
3. 檢查服務狀態 (Rust Backend, Python Service, Flutter App)
4. 分析涉及的 Layers 與影響範圍
5. 決定後續 Phase 需要的 Agents

**輸出**: 影響分析報告 (Layer/模組/影響程度/負責 Agent)

---

## Phase 2: 文件同步

**Lead: system-architect | Members: data-processor, flutter-agent, llm-researcher**

並行分工：
- `system-architect`: ARCHITECTURE.md、跨層 API 契約一致性
- `data-processor`: DATABASE.md vs migration schema
- `flutter-agent`: FEATURES.md vs flutter_app/lib/features/
- `llm-researcher`: LLM/Prompt 文件同步

不一致處：更新文件或回報 Phase 5。

---

## Phase 3: 程式碼檢查

**Lead: qa-engineer | Members: rust-developer, python-developer, flutter-agent, llm-researcher, security-engineer**

並行檢查：
```bash
# rust-developer
cd backend && cargo check && cargo clippy -- -D warnings && cargo fmt --check

# python-developer
cd llm && ruff check app/ tests/

# flutter-agent
cd flutter_app && flutter analyze

# llm-researcher: Prompt 模板品質、Token 效率、Fallback 配置

# security-engineer: cargo audit, bandit -r app/, Prompt 注入檢查
```

問題分類：Error (必須修復) / Warning (建議修復) / Info (可選改進)

---

## Phase 4: 測試執行

**Lead: qa-engineer | Members: rust-developer, python-developer, flutter-agent, llm-researcher, security-engineer**

並行測試：
```bash
# rust-developer
cd backend && cargo test

# python-developer (使用 Anaconda 環境)
cd llm && /c/Users/User/anaconda3/envs/BasiliskCard/python.exe -m pytest tests/ -v --cov=app

# flutter-agent
cd flutter_app && flutter test --coverage
```

**Quality Gates:**
| Gate | 目標 |
|------|------|
| 程式碼品質 | Lint 全通過 |
| 單元測試 | 覆蓋率 > 85% |
| 整合測試 | 全通過 |
| 安全掃描 | 無高危 |
| 效能達標 | P95 < 200ms |
| LLM 品質 | 正確率 > 95% |

---

## Phase 5: 問題修復

**Lead: system-architect | Members: 依問題動態分派**

問題分派：
| 問題類型 | Agent |
|---------|-------|
| Rust/API | rust-developer |
| Python/LLM | python-developer |
| Flutter/UI | flutter-agent |
| Prompt/AI | llm-researcher |
| 資料庫 | data-processor |
| 安全 | security-engineer |
| 跨層 | system-architect 協調 |

**跨層修復順序**: DB → Backend → LLM → Frontend

---

## Phase 5.5: Memory Bank 更新

**Lead: qa-engineer | Members: llm-researcher**

觸發條件：Phase 5 有任何修復。

記錄到 `.claude/projects/.../memory/`:
- `learnings.md` - 問題與解決方案
- `patterns.md` - 程式碼模式
- `decisions.md` - 架構決策

---

## Phase 6: 最終驗證

**Lead: qa-engineer**

1. 重新執行所有測試
2. 逐一驗證 Quality Gates
3. LLM 品質確認 (正確率 > 95%, 一致性 > 90%, 幻覺率 < 5%)
4. 生成最終報告

**驗收結論**: ✅ 通過 | ⚠️ 條件通過 (僅 Low 未解決) | ❌ 未通過

---

## Phase 7: 服務部署

**Lead: system-admin**

觸發條件：有程式碼修改

1. 偵測修改類型 (.rs/.toml → Rust 重編譯, .py → Python 重啟, .dart → Flutter 熱重載)
2. 編譯/打包
3. 背景啟動服務
4. Health Check 驗證

---

## 模式：測試 (`/workflow test`)

**主導: qa-engineer**

```
/workflow test [all|db|backend|frontend|integration] [--fix] [--fix-interactive] [--dry-run]
```

| Scope | Agent |
|-------|-------|
| `backend` | rust-developer: `cargo test` |
| `frontend` | flutter-agent: `flutter test` |
| `db` | data-processor: schema 驗證 |
| `integration` | qa-engineer 協調 |
| `all` | 全部 Agent 並行 |

**失敗處理**: qa-engineer 分派對應 Agent 分析修復。

---

## 模式：程式碼審查 (`/workflow review`)

**主導: qa-engineer | 安全: security-engineer (必執行)**

```
/workflow review [files|branch] [--security-only] [--backend-only] [--frontend-only] [--quick]
```

自動偵測變更檔案類型分派：
- `*.dart` → flutter-agent
- `*.rs` → rust-developer
- `*.py` → python-developer + llm-researcher
- `*.sql` → data-processor
- 全部 → security-engineer (安全必審)

**審查清單**:
- 安全: JWT, 密碼策略, SQL 注入, XSS, CSRF, 敏感資料, Rate Limiting, CORS
- Rust: clippy, 錯誤處理, RESTful, sqlx 參數化, 無 unwrap()
- Python: ruff, 類型提示, Pydantic, Prompt 注入, Token 優化
- Flutter: analyze, 分層架構, BLoC 狀態管理, const Widget, ListView.builder
- 測試: 覆蓋率 > 85%, 邊界條件, 錯誤路徑
- DB: 正規化, 外鍵, 索引, RLS

**嚴重度**: CRITICAL (阻止合併) → HIGH (必須修復) → MEDIUM (建議) → LOW (可選)

---

## 模式：安全審計 (`/workflow security`)

**主導: security-engineer**

```
/workflow security [code|deps|config|infra|db|llm|full] [--quick] [--fix]
```

審計流程：
1. 範圍確認與初始化
2. 程式碼安全分析 (OWASP Top 10, 密鑰洩露, 靜態分析)
3. 依賴漏洞掃描 (`cargo audit`, `pip-audit`, `flutter pub outdated`)
4. 資料庫安全 (RLS, 權限, 加密)
5. LLM 安全 (Prompt 注入, 輸出過濾, Token 限制)
6. 基礎設施安全 (Docker, TLS, 網路策略)
7. 問題修復 (如 --fix)
8. 報告生成

**嚴重度處理**: CRITICAL=立即 | HIGH=24h | MEDIUM=本週 | LOW=計畫性

---

## 模式：文件生成 (`/workflow docs`)

```
/workflow docs [api|arch|db|frontend|llm|security|all]
```

| Type | Agent | 輸出 |
|------|-------|------|
| `api` | rust-developer + python-developer | OpenAPI/Swagger |
| `arch` | system-architect | ARCHITECTURE.md, Mermaid 圖 |
| `db` | data-processor | DATABASE.md, ERD |
| `frontend` | flutter-agent | FEATURES.md, Widget 文件 |
| `llm` | llm-researcher | Prompt 模板, Token 優化指南 |
| `security` | security-engineer | SECURITY.md, RLS 策略 |

---

## 模式：部署 (`/workflow deploy`)

```
/workflow deploy [dev|staging|production]
```

| Phase | 動作 | Agent |
|-------|------|-------|
| 1. 前檢查 | 測試+安全+架構 | qa, security, system-architect |
| 2. 建構 | 編譯打包 | rust, python, flutter |
| 3. 部署 | Docker/K8s | system-admin |
| 4. 驗證 | Health Check + 煙霧測試 | system-admin + qa |

**回滾**: `kubectl rollout undo` 或 `docker-compose down && up`

---

## 模式：開發環境設置 (`/workflow setup`)

```
/workflow setup [all|rust|python|flutter|db]
```

1. **系統檢查** (system-admin): rustc, python, flutter 版本
2. **環境設置** (並行):
   - rust-developer: `cd backend && cargo build && cp .env.example .env`
   - python-developer: `cd llm && pip install -r requirements.txt`
   - flutter-agent: `cd flutter_app && flutter pub get && flutter pub run build_runner build`
3. **資料庫** (data-processor): PostgreSQL migration
4. **驗證** (qa-engineer): 各層測試通過

---

## 模式：LLM 開發 (`/workflow llm`)

**主導: llm-researcher**

```
/workflow llm <需求>
/workflow fix llm <問題>
```

| 問題類型 | 關鍵字 | 主導 | 協作 |
|---------|--------|------|------|
| Prompt | prompt, 輸出, 品質 | llm-researcher | qa-engineer |
| Token | token, 超限, 成本 | llm-researcher | - |
| 供應商 | openai, timeout, fallback | llm-researcher | python-developer |

**品質目標**: 回應時間 < 5s, Token 效率 -30%, 正確率 > 95%, 幻覺率 < 5%

---

## 快速修復 Agent 選擇

| 問題關鍵字 | Agent |
|-----------|-------|
| flutter, ui, widget | flutter-agent |
| rust, api, backend | rust-developer |
| python, llm | python-developer |
| llm, prompt, ai, token | llm-researcher |
| test, coverage, quality | qa-engineer |
| security, 安全 | security-engineer |
| database, sql, migration | rust-developer + data-processor |

---

## 報告格式

```
═══════════════════════════════════════════════════════════════
                Workflow 完成報告
═══════════════════════════════════════════════════════════════

測試結果:
| 服務 | 通過 | 失敗 | 通過率 |
|------|------|------|--------|
| Rust | <n> | <n> | <x>% |
| Python | <n> | <n> | <x>% |
| Flutter | <n> | <n> | <x>% |

Quality Gates:
├── Gate 1: 程式碼品質    [✅/❌]
├── Gate 2: 單元測試      [✅/❌]
├── Gate 3: 整合測試      [✅/❌]
├── Gate 4: 安全掃描      [✅/❌]
├── Gate 5: 效能達標      [✅/❌]
└── Gate 6: LLM 品質      [✅/❌]

驗收結論: [✅ 通過 | ⚠️ 條件通過 | ❌ 未通過]
═══════════════════════════════════════════════════════════════
```

---

**執行參數:** $ARGUMENTS
