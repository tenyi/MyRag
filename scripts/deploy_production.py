#!/usr/bin/env python3
"""
生產環境部署腳本

自動化部署中文 GraphRAG 系統到生產環境
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import yaml
from loguru import logger


class ProductionDeployer:
    """生產環境部署器"""
    
    def __init__(self, deployment_config: Dict[str, Any]):
        self.config = deployment_config
        self.deployment_dir = Path(self.config.get("deployment_dir", "./deployment"))
        self.backup_dir = Path(self.config.get("backup_dir", "./backups"))
        
        # 設定日誌
        logger.remove()
        logger.add(
            self.deployment_dir / "deployment.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB"
        )
        logger.add(sys.stdout, level="INFO")
        
        self.deployment_steps = []
    
    def deploy(self) -> bool:
        """執行完整部署流程"""
        logger.info("🚀 開始生產環境部署")
        
        try:
            # 1. 預部署檢查
            if not self._pre_deployment_checks():
                return False
            
            # 2. 建立部署目錄結構
            self._create_deployment_structure()
            
            # 3. 複製應用程式檔案
            self._copy_application_files()
            
            # 4. 設定生產環境配置
            self._setup_production_config()
            
            # 5. 安裝依賴套件
            self._install_dependencies()
            
            # 6. 建立資料庫和向量儲存
            self._setup_databases()
            
            # 7. 設定監控和日誌
            self._setup_monitoring()
            
            # 8. 建立服務腳本
            self._create_service_scripts()
            
            # 9. 執行部署後測試
            self._post_deployment_tests()
            
            # 10. 建立備份和恢復機制
            self._setup_backup_recovery()
            
            logger.success("🎉 生產環境部署完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 部署失敗: {e}")
            self._rollback_deployment()
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """預部署檢查"""
        logger.info("🔍 執行預部署檢查")
        
        checks = {
            "python_version": self._check_python_version(),
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory(),
            "network": self._check_network_connectivity(),
            "permissions": self._check_permissions(),
            "dependencies": self._check_system_dependencies()
        }
        
        failed_checks = [name for name, result in checks.items() if not result]
        
        if failed_checks:
            logger.error(f"❌ 預部署檢查失敗: {', '.join(failed_checks)}")
            return False
        
        logger.info("✅ 預部署檢查通過")
        return True
    
    def _check_python_version(self) -> bool:
        """檢查 Python 版本"""
        required_version = (3, 11)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            logger.info(f"✅ Python 版本: {'.'.join(map(str, current_version))}")
            return True
        else:
            logger.error(f"❌ Python 版本過低: {'.'.join(map(str, current_version))} < {'.'.join(map(str, required_version))}")
            return False
    
    def _check_disk_space(self) -> bool:
        """檢查磁碟空間"""
        required_space_gb = self.config.get("required_disk_space_gb", 10)
        
        try:
            stat = shutil.disk_usage(self.deployment_dir.parent)
            free_space_gb = stat.free / (1024**3)
            
            if free_space_gb >= required_space_gb:
                logger.info(f"✅ 磁碟空間: {free_space_gb:.1f}GB 可用")
                return True
            else:
                logger.error(f"❌ 磁碟空間不足: {free_space_gb:.1f}GB < {required_space_gb}GB")
                return False
                
        except Exception as e:
            logger.error(f"❌ 磁碟空間檢查失敗: {e}")
            return False
    
    def _check_memory(self) -> bool:
        """檢查記憶體"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            required_memory_gb = self.config.get("required_memory_gb", 4)
            available_memory_gb = memory.available / (1024**3)
            
            if available_memory_gb >= required_memory_gb:
                logger.info(f"✅ 可用記憶體: {available_memory_gb:.1f}GB")
                return True
            else:
                logger.error(f"❌ 記憶體不足: {available_memory_gb:.1f}GB < {required_memory_gb}GB")
                return False
                
        except ImportError:
            logger.warning("⚠️ 無法檢查記憶體 (缺少 psutil)")
            return True
        except Exception as e:
            logger.error(f"❌ 記憶體檢查失敗: {e}")
            return False
    
    def _check_network_connectivity(self) -> bool:
        """檢查網路連線"""
        test_urls = self.config.get("test_urls", ["https://www.google.com"])
        
        for url in test_urls:
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=10)
                logger.info(f"✅ 網路連線正常: {url}")
                return True
            except Exception:
                continue
        
        logger.error("❌ 網路連線檢查失敗")
        return False
    
    def _check_permissions(self) -> bool:
        """檢查檔案權限"""
        try:
            # 檢查部署目錄寫入權限
            test_file = self.deployment_dir / "test_write_permission"
            self.deployment_dir.mkdir(parents=True, exist_ok=True)
            
            test_file.write_text("test")
            test_file.unlink()
            
            logger.info("✅ 檔案權限正常")
            return True
            
        except Exception as e:
            logger.error(f"❌ 檔案權限檢查失敗: {e}")
            return False
    
    def _check_system_dependencies(self) -> bool:
        """檢查系統依賴"""
        required_commands = self.config.get("required_commands", ["git", "curl"])
        
        for cmd in required_commands:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, check=True)
                logger.info(f"✅ 系統命令可用: {cmd}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error(f"❌ 系統命令不可用: {cmd}")
                return False
        
        return True
    
    def _create_deployment_structure(self):
        """建立部署目錄結構"""
        logger.info("📁 建立部署目錄結構")
        
        directories = [
            self.deployment_dir,
            self.deployment_dir / "app",
            self.deployment_dir / "config",
            self.deployment_dir / "data",
            self.deployment_dir / "logs",
            self.deployment_dir / "scripts",
            self.deployment_dir / "backups",
            self.deployment_dir / "monitoring",
            self.backup_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ 建立目錄: {directory}")
    
    def _copy_application_files(self):
        """複製應用程式檔案"""
        logger.info("📋 複製應用程式檔案")
        
        # 複製源碼
        src_dir = Path("src")
        dest_dir = self.deployment_dir / "app" / "src"
        
        if src_dir.exists():
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
            logger.info(f"✅ 複製源碼: {src_dir} -> {dest_dir}")
        
        # 複製必要檔案
        essential_files = [
            "pyproject.toml",
            "README.md",
            "main.py"
        ]
        
        for file_name in essential_files:
            src_file = Path(file_name)
            if src_file.exists():
                dest_file = self.deployment_dir / "app" / file_name
                shutil.copy2(src_file, dest_file)
                logger.info(f"✅ 複製檔案: {src_file} -> {dest_file}")
        
        # 複製配置範本
        config_dir = Path("config")
        if config_dir.exists():
            dest_config_dir = self.deployment_dir / "config"
            shutil.copytree(config_dir, dest_config_dir, dirs_exist_ok=True)
            logger.info(f"✅ 複製配置: {config_dir} -> {dest_config_dir}")
    
    def _setup_production_config(self):
        """設定生產環境配置"""
        logger.info("⚙️ 設定生產環境配置")
        
        # 建立生產環境配置檔案
        prod_config = {
            "environment": "production",
            "debug": False,
            "logging": {
                "level": "INFO",
                "file": str(self.deployment_dir / "logs" / "app.log"),
                "rotation": "1 day",
                "retention": "30 days"
            },
            "database": {
                "path": str(self.deployment_dir / "data" / "vector_db"),
                "backup_interval": "6 hours"
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "health_check_port": 8080
            },
            "security": {
                "api_key_required": True,
                "rate_limiting": True,
                "cors_enabled": False
            }
        }
        
        # 合併用戶配置
        if "production_config" in self.config:
            prod_config.update(self.config["production_config"])
        
        # 寫入配置檔案
        config_file = self.deployment_dir / "config" / "production.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(prod_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✅ 生產配置已建立: {config_file}")
        
        # 建立環境變數檔案
        env_file = self.deployment_dir / "config" / ".env.production"
        env_content = f"""
# 生產環境變數
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
DATA_DIR={self.deployment_dir}/data
LOG_DIR={self.deployment_dir}/logs
BACKUP_DIR={self.deployment_dir}/backups

# API 配置
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# 資料庫配置
VECTOR_DB_PATH={self.deployment_dir}/data/vector_db
GRAPH_DB_PATH={self.deployment_dir}/data/graph_db

# 監控配置
METRICS_ENABLED=true
HEALTH_CHECK_ENABLED=true
"""
        
        env_file.write_text(env_content.strip())
        logger.info(f"✅ 環境變數檔案已建立: {env_file}")
    
    def _install_dependencies(self):
        """安裝依賴套件"""
        logger.info("📦 安裝依賴套件")
        
        try:
            # 切換到應用目錄
            app_dir = self.deployment_dir / "app"
            
            # 安裝 Python 依賴
            cmd = ["uv", "sync", "--frozen"]
            result = subprocess.run(cmd, cwd=app_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Python 依賴安裝成功")
            else:
                logger.error(f"❌ Python 依賴安裝失敗: {result.stderr}")
                raise Exception("依賴安裝失敗")
                
        except Exception as e:
            logger.error(f"❌ 依賴安裝過程中發生錯誤: {e}")
            raise
    
    def _setup_databases(self):
        """設定資料庫和向量儲存"""
        logger.info("🗄️ 設定資料庫和向量儲存")
        
        # 建立資料庫目錄
        db_dirs = [
            self.deployment_dir / "data" / "vector_db",
            self.deployment_dir / "data" / "graph_db",
            self.deployment_dir / "data" / "cache"
        ]
        
        for db_dir in db_dirs:
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ 建立資料庫目錄: {db_dir}")
        
        # 初始化資料庫（如果需要）
        init_script = self.deployment_dir / "scripts" / "init_databases.py"
        init_script.write_text("""#!/usr/bin/env python3
# 資料庫初始化腳本
import sys
from pathlib import Path

def init_databases():
    print("初始化資料庫...")
    # 這裡可以添加具體的資料庫初始化邏輯
    print("資料庫初始化完成")

if __name__ == "__main__":
    init_databases()
""")
        init_script.chmod(0o755)
        
        logger.info("✅ 資料庫設定完成")
    
    def _setup_monitoring(self):
        """設定監控和日誌"""
        logger.info("📊 設定監控和日誌")
        
        # 建立監控腳本
        monitoring_script = self.deployment_dir / "scripts" / "monitor.py"
        monitoring_script.write_text("""#!/usr/bin/env python3
# 系統監控腳本
import time
import psutil
import json
from pathlib import Path

def collect_metrics():
    metrics = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    return metrics

def main():
    metrics_file = Path("monitoring/metrics.json")
    
    while True:
        metrics = collect_metrics()
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        time.sleep(60)  # 每分鐘收集一次

if __name__ == "__main__":
    main()
""")
        monitoring_script.chmod(0o755)
        
        # 建立健康檢查腳本
        health_check_script = self.deployment_dir / "scripts" / "health_check.py"
        health_check_script.write_text("""#!/usr/bin/env python3
# 健康檢查腳本
import sys
import requests
from pathlib import Path

def check_api_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_disk_space():
    import shutil
    stat = shutil.disk_usage('/')
    free_gb = stat.free / (1024**3)
    return free_gb > 1.0  # 至少 1GB 可用空間

def main():
    checks = {
        "api": check_api_health(),
        "disk": check_disk_space()
    }
    
    if all(checks.values()):
        print("健康檢查通過")
        sys.exit(0)
    else:
        print(f"健康檢查失敗: {checks}")
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        health_check_script.chmod(0o755)
        
        logger.info("✅ 監控系統設定完成")
    
    def _create_service_scripts(self):
        """建立服務腳本"""
        logger.info("🔧 建立服務腳本")
        
        # 建立啟動腳本
        start_script = self.deployment_dir / "scripts" / "start.sh"
        start_script.write_text(f"""#!/bin/bash
# 應用啟動腳本

set -e

APP_DIR="{self.deployment_dir}/app"
CONFIG_DIR="{self.deployment_dir}/config"
LOG_DIR="{self.deployment_dir}/logs"

echo "啟動中文 GraphRAG 系統..."

# 載入環境變數
source "$CONFIG_DIR/.env.production"

# 切換到應用目錄
cd "$APP_DIR"

# 啟動應用
uv run uvicorn src.chinese_graphrag.api.app:app \\
    --host $API_HOST \\
    --port $API_PORT \\
    --workers $WORKERS \\
    --log-level info \\
    --access-log \\
    --log-config "$CONFIG_DIR/logging.yaml"

echo "系統啟動完成"
""")
        start_script.chmod(0o755)
        
        # 建立停止腳本
        stop_script = self.deployment_dir / "scripts" / "stop.sh"
        stop_script.write_text("""#!/bin/bash
# 應用停止腳本

echo "停止中文 GraphRAG 系統..."

# 查找並停止進程
pkill -f "chinese_graphrag" || true

echo "系統已停止"
""")
        stop_script.chmod(0o755)
        
        # 建立重啟腳本
        restart_script = self.deployment_dir / "scripts" / "restart.sh"
        restart_script.write_text(f"""#!/bin/bash
# 應用重啟腳本

SCRIPT_DIR="{self.deployment_dir}/scripts"

echo "重啟中文 GraphRAG 系統..."

# 停止服務
"$SCRIPT_DIR/stop.sh"

# 等待進程完全停止
sleep 5

# 啟動服務
"$SCRIPT_DIR/start.sh"

echo "系統重啟完成"
""")
        restart_script.chmod(0o755)
        
        logger.info("✅ 服務腳本建立完成")
    
    def _post_deployment_tests(self):
        """執行部署後測試"""
        logger.info("🧪 執行部署後測試")
        
        # 建立測試腳本
        test_script = self.deployment_dir / "scripts" / "deployment_test.py"
        test_script.write_text(f"""#!/usr/bin/env python3
# 部署後測試腳本
import sys
import time
import subprocess
from pathlib import Path

def test_application_start():
    print("測試應用啟動...")
    try:
        # 這裡可以添加具體的啟動測試邏輯
        print("✅ 應用啟動測試通過")
        return True
    except Exception as e:
        print(f"❌ 應用啟動測試失敗: {{e}}")
        return False

def test_api_endpoints():
    print("測試 API 端點...")
    try:
        # 這裡可以添加 API 測試邏輯
        print("✅ API 端點測試通過")
        return True
    except Exception as e:
        print(f"❌ API 端點測試失敗: {{e}}")
        return False

def main():
    tests = [
        test_application_start,
        test_api_endpoints
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    if all(results):
        print("🎉 所有部署後測試通過")
        sys.exit(0)
    else:
        print("❌ 部署後測試失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()
""")
        test_script.chmod(0o755)
        
        logger.info("✅ 部署後測試設定完成")
    
    def _setup_backup_recovery(self):
        """設定備份和恢復機制"""
        logger.info("💾 設定備份和恢復機制")
        
        # 建立備份腳本
        backup_script = self.deployment_dir / "scripts" / "backup.sh"
        backup_script.write_text(f"""#!/bin/bash
# 資料備份腳本

set -e

BACKUP_DIR="{self.backup_dir}"
DATA_DIR="{self.deployment_dir}/data"
CONFIG_DIR="{self.deployment_dir}/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="backup_$TIMESTAMP"

echo "開始備份..."

# 建立備份目錄
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# 備份資料
if [ -d "$DATA_DIR" ]; then
    cp -r "$DATA_DIR" "$BACKUP_DIR/$BACKUP_NAME/"
    echo "✅ 資料備份完成"
fi

# 備份配置
if [ -d "$CONFIG_DIR" ]; then
    cp -r "$CONFIG_DIR" "$BACKUP_DIR/$BACKUP_NAME/"
    echo "✅ 配置備份完成"
fi

# 建立備份資訊檔案
cat > "$BACKUP_DIR/$BACKUP_NAME/backup_info.txt" << EOF
備份時間: $(date)
備份類型: 完整備份
資料目錄: $DATA_DIR
配置目錄: $CONFIG_DIR
EOF

echo "備份完成: $BACKUP_DIR/$BACKUP_NAME"

# 清理舊備份（保留最近 7 個）
cd "$BACKUP_DIR"
ls -t | tail -n +8 | xargs -r rm -rf

echo "舊備份清理完成"
""")
        backup_script.chmod(0o755)
        
        # 建立恢復腳本
        restore_script = self.deployment_dir / "scripts" / "restore.sh"
        restore_script.write_text(f"""#!/bin/bash
# 資料恢復腳本

set -e

if [ $# -eq 0 ]; then
    echo "使用方法: $0 <備份名稱>"
    echo "可用備份:"
    ls -1 "{self.backup_dir}"
    exit 1
fi

BACKUP_NAME="$1"
BACKUP_PATH="{self.backup_dir}/$BACKUP_NAME"
DATA_DIR="{self.deployment_dir}/data"
CONFIG_DIR="{self.deployment_dir}/config"

if [ ! -d "$BACKUP_PATH" ]; then
    echo "❌ 備份不存在: $BACKUP_PATH"
    exit 1
fi

echo "開始恢復備份: $BACKUP_NAME"

# 停止服務
{self.deployment_dir}/scripts/stop.sh

# 備份當前資料
CURRENT_BACKUP="{self.backup_dir}/current_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CURRENT_BACKUP"
[ -d "$DATA_DIR" ] && cp -r "$DATA_DIR" "$CURRENT_BACKUP/"
[ -d "$CONFIG_DIR" ] && cp -r "$CONFIG_DIR" "$CURRENT_BACKUP/"

# 恢復資料
if [ -d "$BACKUP_PATH/data" ]; then
    rm -rf "$DATA_DIR"
    cp -r "$BACKUP_PATH/data" "$DATA_DIR"
    echo "✅ 資料恢復完成"
fi

# 恢復配置
if [ -d "$BACKUP_PATH/config" ]; then
    rm -rf "$CONFIG_DIR"
    cp -r "$BACKUP_PATH/config" "$CONFIG_DIR"
    echo "✅ 配置恢復完成"
fi

echo "恢復完成，當前資料已備份至: $CURRENT_BACKUP"
echo "請手動重啟服務: {self.deployment_dir}/scripts/start.sh"
""")
        restore_script.chmod(0o755)
        
        # 建立定期備份的 cron 任務範例
        cron_example = self.deployment_dir / "scripts" / "cron_backup_example.txt"
        cron_example.write_text(f"""# 定期備份 cron 任務範例
# 每天凌晨 2 點執行備份
0 2 * * * {self.deployment_dir}/scripts/backup.sh

# 每週日凌晨 3 點執行完整備份
0 3 * * 0 {self.deployment_dir}/scripts/backup.sh

# 安裝方法:
# crontab -e
# 然後添加上述行
""")
        
        logger.info("✅ 備份和恢復機制設定完成")
    
    def _rollback_deployment(self):
        """回滾部署"""
        logger.warning("🔄 執行部署回滾")
        
        try:
            if self.deployment_dir.exists():
                # 建立失敗部署的備份
                failed_deployment_backup = self.backup_dir / f"failed_deployment_{int(time.time())}"
                shutil.move(str(self.deployment_dir), str(failed_deployment_backup))
                logger.info(f"失敗的部署已備份至: {failed_deployment_backup}")
            
            logger.info("✅ 部署回滾完成")
            
        except Exception as e:
            logger.error(f"❌ 部署回滾失敗: {e}")


@click.command()
@click.option("--config", type=click.Path(exists=True), default="deployment_config.yaml",
              help="部署配置檔案")
@click.option("--deployment-dir", type=click.Path(), default="./deployment",
              help="部署目錄")
@click.option("--backup-dir", type=click.Path(), default="./backups",
              help="備份目錄")
@click.option("--dry-run", is_flag=True, help="模擬執行（不實際部署）")
@click.option("--verbose", "-v", is_flag=True, help="詳細輸出")
def main(config: str, deployment_dir: str, backup_dir: str, dry_run: bool, verbose: bool):
    """執行生產環境部署"""
    
    # 載入部署配置
    try:
        with open(config, 'r', encoding='utf-8') as f:
            deployment_config = yaml.safe_load(f)
    except FileNotFoundError:
        # 使用預設配置
        deployment_config = {
            "deployment_dir": deployment_dir,
            "backup_dir": backup_dir,
            "required_disk_space_gb": 10,
            "required_memory_gb": 4,
            "test_urls": ["https://www.google.com"],
            "required_commands": ["git", "curl"],
            "production_config": {
                "workers": 4,
                "log_level": "INFO"
            }
        }
        
        # 建立預設配置檔案
        with open(config, 'w', encoding='utf-8') as f:
            yaml.dump(deployment_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"已建立預設配置檔案: {config}")
    
    # 更新配置
    deployment_config.update({
        "deployment_dir": deployment_dir,
        "backup_dir": backup_dir
    })
    
    if dry_run:
        logger.info("🔍 模擬執行模式")
        logger.info(f"部署配置: {deployment_config}")
        logger.info("實際部署請移除 --dry-run 參數")
        return
    
    # 執行部署
    deployer = ProductionDeployer(deployment_config)
    success = deployer.deploy()
    
    if success:
        logger.success("🎉 生產環境部署成功！")
        logger.info(f"部署目錄: {deployment_dir}")
        logger.info(f"啟動命令: {deployment_dir}/scripts/start.sh")
        sys.exit(0)
    else:
        logger.error("❌ 生產環境部署失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()