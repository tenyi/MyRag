# 變更記錄

本檔案記錄 Chinese GraphRAG 系統的所有重要變更。

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
