# 變更記錄

本檔案記錄 Chinese GraphRAG 系統的所有重要變更。

## [0.2.3] - 2025/08/02 18:34

### Added

- **GraphRAG API 整合** - 完整整合 Microsoft GraphRAG 官方 API
  - 新增 `_create_graphrag_pipeline_config()` 方法，創建 GraphRAG Pipeline 配置
  - 新增 `_run_graphrag_pipeline()` 方法，使用官方 pipeline 執行索引
  - 新增 `_process_graphrag_results()` 方法，處理 GraphRAG pipeline 結果
  - 新增 `_read_graphrag_output_stats()` 方法，從輸出文件讀取統計資訊
  - 新增 `_run_custom_pipeline()` 方法，作為自定義流程的回退選項

- **GraphRAG Workflow 函數整合** - 使用官方 workflow 函數進行各個處理步驟
  - 新增 `_create_text_units_with_graphrag()` 使用 `create_base_text_units`
  - 新增 `_extract_entities_and_relationships_with_graphrag()` 使用多個實體關係提取 workflows
  - 新增 `_detect_communities_with_graphrag()` 使用社群檢測 workflows
  - 新增資料轉換方法：`_convert_graphrag_entities_to_models()`, `_convert_graphrag_relationships_to_models()`, `_convert_graphrag_communities_to_models()`
  - 新增 `_extract_community_reports()` 方法提取社群報告資訊

### Changed

- **索引流程優化** - 重構 `index_documents()` 方法
  - 優先使用 GraphRAG 官方 pipeline，不可用時回退到自定義流程
  - 改進錯誤處理和日誌記錄
  - 支援 GraphRAG 和自定義流程的無縫切換

- **API 使用改進** - 正確使用之前引用但未使用的 GraphRAG API
  - 使用 `create_pipeline_config` 創建管道配置
  - 使用 `run_pipeline_with_config` 執行完整索引流程
  - 使用所有 workflow 函數：`create_base_text_units`, `create_base_extracted_entities`, `create_summarized_entities`, `create_base_entity_graph`, `create_final_entities`, `create_final_relationships`, `create_final_communities`, `create_final_community_reports`

### Technical

- **架構改進** - 增強 GraphRAGIndexer 類別的功能和可擴展性
- **相容性提升** - 確保與 Microsoft GraphRAG 框架的完全相容
- **效能優化** - 利用 GraphRAG 官方實現提升處理效率
- **錯誤恢復** - 實現優雅的回退機制，確保系統穩定性

## [0.2.2] - 2025/08/02 14:59

### Fixed

- **索引引擎測試修復** - 修復 4 個 GraphRAG 索引引擎測試失敗問題
  - 添加 `pytest-mock` 依賴到開發環境，解決 `mocker` fixture 缺失問題
  - 修復 `test_entity_and_relationship_extraction` 中 `llm_name` 未定義錯誤
  - 修復 `test_detect_communities` 測試的社群檢測流程
  - 修復 `test_create_embeddings` 測試的向量嵌入建立流程
  - 修復 `test_full_indexing_workflow` 測試的完整索引工作流程
  - 改進索引狀態管理，確保測試時正確更新內部狀態

### Technical

- **測試依賴管理** - 在 `pyproject.toml` 中添加 `pytest-mock>=3.10.0` 依賴
- **LLM 配置處理** - 改進 LLM 配置訪問的安全性，使用字典 `.get()` 方法
- **索引狀態同步** - 確保 `index_documents` 方法在使用模擬時也能正確更新索引狀態
- **錯誤處理改進** - 提高索引引擎在測試環境下的穩定性

## [0.2.1] - 2025/08/02 14:48

### Fixed

- **配置系統測試修復** - 修復 `test_create_default_config` 測試失敗問題
  - 修復預設配置中環境變數缺少預設值的問題
  - 更新 `create_default_config` 方法，為 `GRAPHRAG_API_KEY` 環境變數提供預設值
  - 確保在測試環境中能正確載入預設配置檔案
  - 所有配置系統測試現在都能正常通過

### Technical

- **環境變數處理改進** - 預設配置中的環境變數現在使用 `${VAR:default}` 語法
- **測試穩定性提升** - 消除了因環境變數未設定導致的測試失敗
- **配置載入器健壯性** - 提高了配置系統在不同環境下的可靠性

## [0.2.0] - 2025/08/01 17:41

### Added

- **完整文件系統建立** - Task 13 完成
  - 新增安裝和配置指南 (docs/installation_guide.md)
  - 新增 API 使用文件 (docs/api_usage_guide.md)
  - 新增故障排除指南 (docs/troubleshooting_guide.md)
  - 新增架構和設計文件 (docs/architecture_design.md)
  - 新增程式碼貢獻指南 (docs/contributing_guide.md)
  - 新增程式碼範例和教學 (docs/examples_and_tutorials.md)
  - 新增文件導航系統 (docs/README.md, docs/navigation.md)
  - 新增文件驗證報告 (docs/documentation_validation_report.md)

### Changed

- **更新主 README.md** - 添加完整的文件導航連結
- **文件結構優化** - 建立系統化的文件組織架構

### Features

- **使用者文件完整性** - 涵蓋安裝、配置、API 使用和故障排除
- **開發者文件完整性** - 包含架構設計、貢獻指南和程式碼範例
- **多層次導航系統** - 支援不同使用情境的文件導航路徑
- **文件品質保證** - 建立完整的驗證和維護機制

### Technical

- **文件標準化** - 統一的 Markdown 格式和風格規範
- **交叉引用系統** - 完整的文件間連結和引用
- **內容一致性** - 術語、範例和格式的統一標準
- **維護便利性** - 結構化的文件組織便於後續維護

### Infrastructure

- **文件自動化準備** - 為未來的文件自動化檢查做好準備
- **多語言支援準備** - 建立支援多語言版本的基礎架構
- **社群貢獻支援** - 完整的文件貢獻指南和流程

## [0.1.0] - 2025/07/25 之前

### Added

- **核心系統架構** - Tasks 1-12 完成
  - 專案基礎架構和開發環境 (Task 1)
  - 核心資料模型和介面 (Task 2)
  - 中文文本處理模組 (Task 3)
  - 多模型 Embedding 服務 (Task 4)
  - 向量資料庫整合 (Task 5)
  - GraphRAG 核心功能 (Task 6)
  - 查詢和檢索系統 (Task 7)
  - 系統配置和管理功能 (Task 8)
  - 命令列介面和 API (Task 9)
  - 錯誤處理和恢復機制 (Task 10)
  - 測試套件 (Task 11)
  - 效能優化和擴展性改進 (Task 12)

### Features

- **中文優化處理** - 專門針對中文文件的處理能力
- **多模型支援** - 支援多種 LLM 和 Embedding 模型
- **知識圖譜建構** - 自動建構實體關係和社群結構
- **向量化儲存** - 持久化向量資料以提供高效檢索
- **REST API 介面** - 完整的 HTTP API 支援
- **命令列工具** - 豐富的 CLI 功能
- **效能優化** - 智慧快取、批次處理和成本控制

### Technical

- **Python 3.11+** - 現代 Python 開發環境
- **uv 套件管理** - 高效的依賴管理
- **Microsoft GraphRAG** - 基於成熟的 GraphRAG 框架
- **BGE-M3 Embedding** - 中文優化的向量化模型
- **LanceDB 向量資料庫** - 高效的向量儲存和檢索
- **FastAPI** - 現代的 API 框架
- **完整測試覆蓋** - 單元測試、整合測試和效能測試

---

## 版本號規則

本專案使用 [語義化版本](https://semver.org/) 規則：

- **主版本號 (Major)**：不相容的 API 變更
- **次版本號 (Minor)**：向後相容的功能新增
- **修訂版本號 (Patch)**：向後相容的錯誤修復

## 變更類型

- **Added**: 新增功能
- **Changed**: 現有功能的變更
- **Deprecated**: 即將移除的功能
- **Removed**: 已移除的功能
- **Fixed**: 錯誤修復
- **Security**: 安全性相關變更
- **Infrastructure**: 基礎設施和工具變更
- **Features**: 主要功能特性
- **Technical**: 技術實作細節
