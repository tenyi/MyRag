# 程式碼風格與慣例

## 命名規範
- **模組檔案**: snake_case (`chinese_text_processor.py`)
- **類別名稱**: PascalCase (`ChineseTextProcessor`)
- **函數名稱**: snake_case (`extract_entities`)
- **常數**: UPPER_SNAKE_CASE (`DEFAULT_BATCH_SIZE`)
- **變數**: snake_case

## 程式碼風格
- **行長度**: 88 字元（black 標準）
- **型別提示**: 必須使用完整的型別提示
- **註解語言**: 使用繁體中文註解和文件字串
- **測試覆蓋率**: 目標 >90%

## 文件字串格式
使用中文描述功能和用途：
```python
def process_text(text: str) -> List[str]:
    \"\"\"處理中文文本。
    
    Args:
        text: 待處理的中文文本
        
    Returns:
        處理後的文本片段列表
        
    Raises:
        ValueError: 當輸入文本為空時
    \"\"\"
```

## 錯誤處理
定義了具體的異常類型：
- `DocumentProcessingError`: 文件處理相關錯誤
- `EmbeddingServiceError`: Embedding 服務錯誤  
- `DatabaseError`: 資料庫相關錯誤

## 測試結構
- 測試檔案鏡像 src 結構
- 測試類別使用 `Test` 前綴
- 每個模組/功能都有對應的測試檔案
- 特別關注中文文本處理測試

## 專案配置工具
- **Black**: 程式碼格式化，行長度 88
- **isort**: import 排序，配置與 black 相容
- **mypy**: 型別檢查，嚴格模式
- **pytest**: 測試框架，包含覆蓋率報告