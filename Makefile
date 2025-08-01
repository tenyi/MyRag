# Chinese GraphRAG 測試自動化 Makefile

.PHONY: help install test test-unit test-integration test-chinese test-performance
.PHONY: quality-check security-scan coverage report clean setup-test-data
.PHONY: test-automation ci-local

# 預設目標
help:
	@echo "Chinese GraphRAG 測試自動化命令"
	@echo ""
	@echo "安裝和設定:"
	@echo "  install           安裝專案依賴"
	@echo "  setup-test-data   初始化測試資料"
	@echo ""
	@echo "測試執行:"
	@echo "  test              執行所有測試"
	@echo "  test-unit         執行單元測試"
	@echo "  test-integration  執行整合測試"
	@echo "  test-chinese      執行中文功能測試"
	@echo "  test-performance  執行效能測試"
	@echo ""
	@echo "品質檢查:"
	@echo "  quality-check     執行程式碼品質檢查"
	@echo "  security-scan     執行安全性掃描"
	@echo "  coverage          生成覆蓋率報告"
	@echo ""
	@echo "報告和自動化:"
	@echo "  report            生成測試報告"
	@echo "  test-automation   執行完整測試自動化管道"
	@echo "  ci-local          本地模擬 CI 環境"
	@echo ""
	@echo "清理:"
	@echo "  clean             清理生成的檔案"

# 安裝依賴
install:
	@echo "📦 安裝專案依賴..."
	uv sync --dev

# 初始化測試資料
setup-test-data:
	@echo "🚀 初始化測試資料..."
	uv run python scripts/init_test_data.py --reset

# 執行所有測試
test:
	@echo "🧪 執行所有測試..."
	uv run pytest tests/ \
		--cov=src/chinese_graphrag \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing \
		--junit-xml=pytest-all.xml \
		--tb=short \
		-v

# 執行單元測試
test-unit:
	@echo "🧪 執行單元測試..."
	uv run pytest tests/ \
		--cov=src/chinese_graphrag \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing \
		--junit-xml=pytest-unit.xml \
		-m "not integration and not slow" \
		--tb=short \
		-v

# 執行整合測試
test-integration:
	@echo "🧪 執行整合測試..."
	uv run pytest tests/integration/ \
		--junit-xml=pytest-integration.xml \
		-m "integration" \
		--tb=short \
		-v \
		--timeout=300

# 執行中文功能測試
test-chinese:
	@echo "🧪 執行中文功能測試..."
	uv run pytest tests/ \
		--junit-xml=pytest-chinese.xml \
		-m "chinese" \
		--tb=short \
		-v

# 執行效能測試
test-performance:
	@echo "⚡ 執行效能測試..."
	uv run pytest tests/integration/test_performance.py \
		--benchmark-json=benchmark.json \
		--benchmark-only \
		-v

# 程式碼品質檢查
quality-check:
	@echo "🔍 執行程式碼品質檢查..."
	@echo "  檢查程式碼格式 (Black)..."
	uv run black --check --diff src/ tests/
	@echo "  檢查導入排序 (isort)..."
	uv run isort --check-only --diff src/ tests/
	@echo "  檢查程式碼風格 (flake8)..."
	uv run flake8 src/ tests/
	@echo "  檢查型別註解 (mypy)..."
	uv run mypy src/ || true

# 安全性掃描
security-scan:
	@echo "🔒 執行安全性掃描..."
	pip install bandit[toml]
	bandit -r src/ -f json -o bandit-report.json || true
	bandit -r src/ -f txt

# 生成覆蓋率報告
coverage: test-unit
	@echo "📊 生成覆蓋率報告..."
	@echo "HTML 報告已生成在 htmlcov/ 目錄"
	@echo "XML 報告已生成: coverage.xml"

# 生成測試報告
report:
	@echo "📊 生成測試報告..."
	uv run python scripts/generate_test_report.py \
		--output-dir test-reports \
		--format html,json,summary \
		--include-coverage \
		--include-performance

# 執行完整測試自動化管道
test-automation:
	@echo "🚀 執行完整測試自動化管道..."
	uv run python scripts/run_test_automation.py \
		--test-types unit integration chinese

# 本地模擬 CI 環境
ci-local: install setup-test-data quality-check test security-scan report
	@echo "🎯 執行品質閘門檢查..."
	uv run python scripts/quality_gate.py \
		--test-results-dir . \
		--coverage-file coverage.xml \
		--benchmark-file benchmark.json \
		--security-file bandit-report.json \
		--config config/quality_gate.yaml

# 清理生成的檔案
clean:
	@echo "🧹 清理生成的檔案..."
	rm -rf htmlcov/
	rm -rf test-reports/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf tests/__pycache__/
	rm -rf src/**/__pycache__/
	rm -f coverage.xml
	rm -f pytest-*.xml
	rm -f benchmark.json
	rm -f bandit-report.json
	rm -f quality-gate-results.json
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 格式化程式碼
format:
	@echo "✨ 格式化程式碼..."
	uv run black src/ tests/
	uv run isort src/ tests/

# 快速測試（僅單元測試）
test-quick:
	@echo "⚡ 執行快速測試..."
	uv run pytest tests/ \
		-m "not integration and not slow" \
		--tb=short \
		-x \
		-q

# 監視模式測試
test-watch:
	@echo "👀 監視模式測試..."
	uv run pytest-watch tests/ \
		-m "not integration and not slow" \
		--tb=short

# 檢查依賴更新
check-deps:
	@echo "🔍 檢查依賴更新..."
	uv tree --outdated

# 更新依賴
update-deps:
	@echo "📦 更新依賴..."
	uv sync --upgrade

# 建立開發環境
dev-setup: install setup-test-data
	@echo "🛠️ 開發環境設定完成！"
	@echo ""
	@echo "可用命令："
	@echo "  make test-quick    # 快速測試"
	@echo "  make test          # 完整測試"
	@echo "  make quality-check # 品質檢查"
	@echo "  make format        # 格式化程式碼"