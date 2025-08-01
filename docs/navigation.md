# 文件導航配置

本文件定義了 Chinese GraphRAG 系統文件的導航結構和組織方式。

## 文件結構

```
docs/
├── README.md                    # 文件首頁和導航
├── navigation.md               # 本文件 - 導航配置
├── installation_guide.md       # 安裝和配置指南
├── api_usage_guide.md          # API 使用文件
├── troubleshooting_guide.md    # 故障排除指南
├── architecture_design.md      # 架構和設計文件
├── contributing_guide.md       # 程式碼貢獻指南
├── examples_and_tutorials.md   # 程式碼範例和教學
├── performance_optimization.md # 效能優化指南（已存在）
├── performance_deployment_guide.md # 部署指南（已存在）
└── test_automation.md         # 測試自動化（已存在）
```

## 文件分類

### 1. 使用者文件 (User Documentation)

**目標讀者**: 終端使用者、系統管理員

- **[安裝和配置指南](installation_guide.md)**
  - 系統需求
  - 安裝步驟
  - 環境配置
  - 驗證安裝

- **[API 使用文件](api_usage_guide.md)**
  - API 概述
  - 端點參考
  - 使用範例
  - 錯誤處理

- **[故障排除指南](troubleshooting_guide.md)**
  - 常見問題
  - 診斷工具
  - 解決方案
  - 聯繫支援

### 2. 開發者文件 (Developer Documentation)

**目標讀者**: 開發者、貢獻者

- **[架構和設計文件](architecture_design.md)**
  - 系統架構
  - 核心元件
  - 設計模式
  - 資料模型

- **[程式碼貢獻指南](contributing_guide.md)**
  - 開發環境設定
  - 程式碼規範
  - 提交流程
  - 測試指南

- **[程式碼範例和教學](examples_and_tutorials.md)**
  - 快速開始
  - 基礎教學
  - 進階範例
  - 最佳實踐

### 3. 運維文件 (Operations Documentation)

**目標讀者**: 運維人員、系統管理員

- **[效能優化指南](performance_optimization.md)**
  - 效能調優
  - 監控指標
  - 最佳實踐

- **[部署指南](performance_deployment_guide.md)**
  - 部署架構
  - 環境配置
  - 維護指南

- **[測試自動化](test_automation.md)**
  - 測試策略
  - 自動化流程
  - 品質控制

## 導航路徑

### 新使用者路徑

```
README.md → installation_guide.md → examples_and_tutorials.md → api_usage_guide.md
```

### 開發者路徑

```
README.md → architecture_design.md → contributing_guide.md → examples_and_tutorials.md
```

### 運維人員路徑

```
README.md → installation_guide.md → performance_deployment_guide.md → performance_optimization.md
```

### 問題解決路徑

```
README.md → troubleshooting_guide.md → [相關技術文件]
```

## 交叉引用

### 文件間連結

- 每個文件都應包含相關文件的連結
- 使用相對路徑進行文件間連結
- 在文件末尾提供「相關文件」區段

### 內容交叉引用

- 安裝指南 ↔ 故障排除指南
- API 文件 ↔ 程式碼範例
- 架構文件 ↔ 貢獻指南
- 效能優化 ↔ 部署指南

## 文件維護

### 更新頻率

- **高頻更新**: examples_and_tutorials.md, troubleshooting_guide.md
- **中頻更新**: api_usage_guide.md, installation_guide.md
- **低頻更新**: architecture_design.md, contributing_guide.md

### 版本控制

- 每次重大更新都應更新文件版本
- 在 README.md 中維護更新日誌
- 標記過時或已棄用的內容

### 品質檢查

- 定期檢查連結有效性
- 驗證程式碼範例可執行性
- 確保文件內容與程式碼同步

## 文件標準

### 格式規範

- 使用 Markdown 格式
- 統一的標題層級結構
- 一致的程式碼區塊格式
- 標準化的表格和列表格式

### 內容規範

- 每個文件都有目錄
- 清晰的章節劃分
- 豐富的程式碼範例
- 適當的圖表和說明

### 語言規範

- 使用繁體中文
- 技術術語保持一致
- 友好和專業的語調
- 清晰簡潔的表達

## 搜尋和索引

### 關鍵詞標籤

每個文件都應包含相關的關鍵詞標籤：

- **installation_guide.md**: 安裝, 配置, 環境, 依賴, Python, uv
- **api_usage_guide.md**: API, REST, 端點, 查詢, 索引, HTTP
- **troubleshooting_guide.md**: 錯誤, 問題, 診斷, 解決, 修復
- **architecture_design.md**: 架構, 設計, 模式, 元件, 系統
- **contributing_guide.md**: 貢獻, 開發, 程式碼, 測試, 規範
- **examples_and_tutorials.md**: 範例, 教學, 程式碼, 實作, 使用

### 搜尋優化

- 在文件中使用描述性的標題
- 包含常見的搜尋關鍵詞
- 提供同義詞和替代表達
- 建立詞彙表和術語對照

## 多語言支援

### 當前語言

- 主要語言: 繁體中文 (zh-TW)
- 程式碼註解: 繁體中文
- 變數命名: 英文

### 未來擴展

- 簡體中文 (zh-CN)
- 英文 (en-US)
- 日文 (ja-JP)

## 文件工具

### 建議工具

- **編輯器**: VS Code + Markdown 擴展
- **預覽**: Markdown Preview Enhanced
- **連結檢查**: markdown-link-check
- **格式化**: Prettier
- **拼寫檢查**: Code Spell Checker

### 自動化

- 使用 GitHub Actions 進行文件檢查
- 自動生成目錄和索引
- 連結有效性檢查
- 文件同步檢查

這個導航配置文件將幫助維護者和貢獻者更好地組織和維護文件系統。
