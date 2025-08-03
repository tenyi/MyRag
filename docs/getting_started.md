# 入門指南

歡迎使用 Chinese GraphRAG 系統！本指南將引導您完成從零開始設定和執行您的第一個中文 RAG 查詢的完整過程。

## 1. 專案介紹

Chinese GraphRAG 是一個基於 Microsoft GraphRAG 框架的知識圖譜檢索增強生成（RAG）系統，專為處理中文內容而優化。它能將您的文件轉換為知識圖譜，並透過大型語言模型（LLM）提供智慧問答能力。

## 2. 環境準備

在開始之前，請確保您的系統已安裝以下軟體：

- **Python 3.11+**
- **Git**
- 一個終端機或命令列工具

## 3. 安裝與設定

### 第一步：取得程式碼

首先，使用 `git` 將專案複製到您的本機電腦：

```bash
git clone https://github.com/your-org/chinese-graphrag.git
cd chinese-graphrag
```

### 第二步：安裝依賴套件

本專案使用 `uv` 作為套件管理工具，它是一個非常快速的 Python 套件安裝程式和解析器。

1.  **安裝 uv** (如果您尚未安裝):
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    安裝後請重啟您的終端機。

2.  **安裝專案依賴**:
    在專案根目錄下，執行以下命令來安裝所有必要的套件：
    ```bash
    uv sync
    ```
    此命令會讀取 `pyproject.toml` 文件並安裝所有指定的依賴，這可能需要幾分鐘的時間。

### 第三步：環境設定

專案的設定是透過 `.env` 檔案和 `config/settings.yaml` 來管理的。

1.  **建立 `.env` 檔案**:
    複製範本檔案以建立您自己的本地環境設定：
    ```bash
    cp .env.example .env
    ```
    接著，用文字編輯器打開 `.env` 檔案。您需要至少設定一個大型語言模型（LLM）的 API 金鑰。例如，如果您使用 OpenAI：
    ```
    # .env

    # OpenAI API 配置
    GRAPHRAG_API_KEY="sk-YourOpenAI_API_Key_Here"
    ```
    將 `sk-YourOpenAI_API_Key_Here` 替換為您自己的金鑰。

2.  **檢查 `settings.yaml`**:
    同樣地，複製設定檔範本：
    ```bash
    cp config/settings.yaml.example config/settings.yaml
    ```
    對於初次使用，預設的 `settings.yaml` 內容通常無需修改。它預設使用 OpenAI 的 `gpt-4.1-mini` 和 `text-embedding-3-small` 模型，並將所有資料儲存在本地的 `./data` 目錄中。

## 4. 執行您的第一個 RAG 流程

現在，一切準備就緒！讓我們來執行一個完整的 RAG 流程，包括索引文件和進行查詢。

### 第一步：準備您的文件

1.  在專案根目錄下，建立一個名為 `documents` 的資料夾。
    ```bash
    mkdir documents
    ```
2.  在 `documents` 資料夾中，建立一個名為 `hello.txt` 的文字檔案，並貼上以下內容：
    ```txt
    Chinese GraphRAG 是一個強大的工具。它專為中文優化，可以幫助使用者從文件中提取知識並建立知識圖譜。
    ```

### 第二步：索引文件

索引是系統讀取您的文件、理解內容、提取實體和關係，並將它們轉換為向量和圖譜結構的過程。

執行以下命令來索引 `documents` 資料夾中的所有文件：

```bash
uv run chinese-graphrag index --input ./documents --output ./data
```

- `--input ./documents`: 指定包含您文件的資料夾。
- `--output ./data`: 指定儲存索引結果的位置。

您會看到系統開始處理文件，並在完成後顯示成功訊息。

### 第三步：進行查詢

索引完成後，您就可以開始問問題了！

執行以下命令來進行查詢：

```bash
uv run chinese-graphrag query "Chinese GraphRAG 是什麼？"
```

系統將會：
1.  理解您的問題。
2.  在剛剛建立的索引中搜索最相關的資訊。
3.  使用 LLM 整合這些資訊並生成一個自然語言的回答。

您應該會看到類似以下的輸出：

```
> 正在執行查詢：Chinese GraphRAG 是什麼？
> 查詢結果：
Chinese GraphRAG 是一個專為中文優化的強大工具，可以幫助使用者從文件中提取知識並建立知識圖譜。
> 引用來源：
- hello.txt (100.0%)
```

恭喜！您已經成功地使用 Chinese GraphRAG 完成了一次從文件到問答的完整流程。

## 5. 接下來呢？

現在您已經掌握了基本操作，可以嘗試以下進階功能：

- **索引您自己的文件**：將您自己的 `.txt`, `.pdf`, `.md` 或 `.docx` 文件放入 `documents` 資料夾中，然後重新執行索引命令。
- **嘗試不同的問題**：對您的文件提出各種問題，測試系統的理解能力。
- **探索 API**：透過 `uv run chinese-graphrag api server` 啟動 API 服務，並在瀏覽器中打開 `http://localhost:8000/docs` 來探索互動式的 API 文件。
- **查閱其他文件**：本專案的 `docs` 目錄中有更多詳細的文件，涵蓋架構設計、API 使用指南等。

如果您遇到任何問題，請隨時查閱 [故障排除指南](./troubleshooting_guide.md) 或在專案的 GitHub Issues 中提出問題。
