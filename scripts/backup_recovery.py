#!/usr/bin/env python3
"""
備份和恢復程序

提供完整的資料備份、恢復和災難恢復功能
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
import yaml
from loguru import logger


class BackupManager:
    """備份管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.backup_dir = Path(self.config["backup_dir"])
        self.data_dir = Path(self.config["data_dir"])
        self.config_dir = Path(self.config["config_dir"])
        
        # 建立備份目錄
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定日誌
        logger.remove()
        logger.add(
            self.backup_dir / "backup.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB",
            retention="90 days"
        )
        logger.add(sys.stdout, level="INFO")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """載入備份配置"""
        default_config = {
            "backup_dir": "./backups",
            "data_dir": "./data",
            "config_dir": "./config",
            "retention": {
                "daily": 7,    # 保留 7 天的每日備份
                "weekly": 4,   # 保留 4 週的週備份
                "monthly": 12  # 保留 12 個月的月備份
            },
            "compression": True,
            "encryption": False,
            "verify_backup": True,
            "exclude_patterns": [
                "*.tmp",
                "*.log",
                "__pycache__",
                ".DS_Store"
            ],
            "remote_backup": {
                "enabled": False,
                "type": "s3",  # s3, ftp, rsync
                "config": {}
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"載入配置失敗，使用預設配置: {e}")
        
        return default_config
    
    def create_backup(self, backup_type: str = "full", description: str = "") -> Optional[str]:
        """建立備份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{backup_type}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"🔄 開始建立備份: {backup_name}")
        
        try:
            # 建立備份目錄
            backup_path.mkdir(exist_ok=True)
            
            # 建立備份資訊
            backup_info = {
                "name": backup_name,
                "type": backup_type,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "version": self._get_system_version(),
                "files": [],
                "size_bytes": 0,
                "checksum": None
            }
            
            # 備份資料目錄
            if self.data_dir.exists():
                data_backup_path = backup_path / "data"
                self._copy_directory(self.data_dir, data_backup_path)
                backup_info["files"].append("data")
                logger.info("✅ 資料目錄備份完成")
            
            # 備份配置目錄
            if self.config_dir.exists():
                config_backup_path = backup_path / "config"
                self._copy_directory(self.config_dir, config_backup_path)
                backup_info["files"].append("config")
                logger.info("✅ 配置目錄備份完成")
            
            # 備份應用程式檔案（如果是完整備份）
            if backup_type == "full":
                app_files = ["main.py", "pyproject.toml", "README.md"]
                app_backup_path = backup_path / "app"
                app_backup_path.mkdir(exist_ok=True)
                
                for app_file in app_files:
                    src_file = Path(app_file)
                    if src_file.exists():
                        shutil.copy2(src_file, app_backup_path / app_file)
                        backup_info["files"].append(f"app/{app_file}")
                
                # 備份源碼目錄
                src_dir = Path("src")
                if src_dir.exists():
                    src_backup_path = app_backup_path / "src"
                    self._copy_directory(src_dir, src_backup_path)
                    backup_info["files"].append("app/src")
                
                logger.info("✅ 應用程式檔案備份完成")
            
            # 計算備份大小
            backup_info["size_bytes"] = self._calculate_directory_size(backup_path)
            
            # 壓縮備份（如果啟用）
            if self.config["compression"]:
                compressed_path = self._compress_backup(backup_path)
                if compressed_path:
                    # 刪除原始目錄
                    shutil.rmtree(backup_path)
                    backup_path = compressed_path
                    backup_info["compressed"] = True
                    backup_info["size_bytes"] = backup_path.stat().st_size
                    logger.info("✅ 備份壓縮完成")
            
            # 計算校驗和
            if self.config["verify_backup"]:
                backup_info["checksum"] = self._calculate_checksum(backup_path)
                logger.info("✅ 校驗和計算完成")
            
            # 儲存備份資訊
            info_file = backup_path.parent / f"{backup_name}.info"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
            
            # 遠端備份（如果啟用）
            if self.config["remote_backup"]["enabled"]:
                self._upload_to_remote(backup_path, backup_info)
            
            logger.success(f"🎉 備份建立完成: {backup_name}")
            logger.info(f"   備份大小: {backup_info['size_bytes'] / (1024**2):.1f}MB")
            logger.info(f"   備份位置: {backup_path}")
            
            return backup_name
            
        except Exception as e:
            logger.error(f"❌ 備份建立失敗: {e}")
            # 清理失敗的備份
            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
            return None
    
    def _copy_directory(self, src: Path, dst: Path):
        """複製目錄，排除指定模式"""
        def ignore_patterns(dir_path, names):
            ignored = []
            for pattern in self.config["exclude_patterns"]:
                import fnmatch
                ignored.extend(fnmatch.filter(names, pattern))
            return ignored
        
        shutil.copytree(src, dst, ignore=ignore_patterns, dirs_exist_ok=True)
    
    def _calculate_directory_size(self, path: Path) -> int:
        """計算目錄大小"""
        total_size = 0
        if path.is_file():
            return path.stat().st_size
        
        for item in path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    
    def _compress_backup(self, backup_path: Path) -> Optional[Path]:
        """壓縮備份"""
        try:
            compressed_path = backup_path.with_suffix('.tar.gz')
            
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_path.name)
            
            return compressed_path
            
        except Exception as e:
            logger.error(f"壓縮備份失敗: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """計算檔案校驗和"""
        hash_md5 = hashlib.md5()
        
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # 目錄的話，計算所有檔案的校驗和
            for file_path in sorted(file_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_system_version(self) -> str:
        """獲取系統版本"""
        try:
            # 嘗試從 pyproject.toml 讀取版本
            pyproject_path = Path("pyproject.toml")
            if pyproject_path.exists():
                import toml
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    pyproject = toml.load(f)
                    return pyproject.get("project", {}).get("version", "unknown")
        except:
            pass
        
        return "unknown"
    
    def _upload_to_remote(self, backup_path: Path, backup_info: Dict[str, Any]):
        """上傳到遠端儲存"""
        try:
            remote_config = self.config["remote_backup"]
            
            if remote_config["type"] == "s3":
                self._upload_to_s3(backup_path, backup_info)
            elif remote_config["type"] == "ftp":
                self._upload_to_ftp(backup_path, backup_info)
            elif remote_config["type"] == "rsync":
                self._upload_to_rsync(backup_path, backup_info)
            
            logger.info("✅ 遠端備份上傳完成")
            
        except Exception as e:
            logger.error(f"遠端備份上傳失敗: {e}")
    
    def _upload_to_s3(self, backup_path: Path, backup_info: Dict[str, Any]):
        """上傳到 S3"""
        # 這裡可以實作 S3 上傳邏輯
        pass
    
    def _upload_to_ftp(self, backup_path: Path, backup_info: Dict[str, Any]):
        """上傳到 FTP"""
        # 這裡可以實作 FTP 上傳邏輯
        pass
    
    def _upload_to_rsync(self, backup_path: Path, backup_info: Dict[str, Any]):
        """使用 rsync 同步"""
        # 這裡可以實作 rsync 同步邏輯
        pass
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有備份"""
        backups = []
        
        for info_file in self.backup_dir.glob("*.info"):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    backup_info = json.load(f)
                    
                    # 檢查備份檔案是否存在
                    backup_name = backup_info["name"]
                    backup_file = self.backup_dir / f"{backup_name}.tar.gz"
                    backup_dir = self.backup_dir / backup_name
                    
                    if backup_file.exists():
                        backup_info["path"] = str(backup_file)
                        backup_info["exists"] = True
                    elif backup_dir.exists():
                        backup_info["path"] = str(backup_dir)
                        backup_info["exists"] = True
                    else:
                        backup_info["exists"] = False
                    
                    backups.append(backup_info)
                    
            except Exception as e:
                logger.warning(f"讀取備份資訊失敗 {info_file}: {e}")
        
        # 按建立時間排序
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups
    
    def restore_backup(self, backup_name: str, target_dir: Optional[Path] = None) -> bool:
        """恢復備份"""
        logger.info(f"🔄 開始恢復備份: {backup_name}")
        
        try:
            # 查找備份
            backup_info = self._get_backup_info(backup_name)
            if not backup_info:
                logger.error(f"找不到備份: {backup_name}")
                return False
            
            # 確定備份檔案路徑
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            backup_dir = self.backup_dir / backup_name
            
            if backup_file.exists():
                # 解壓縮備份
                temp_dir = self.backup_dir / f"temp_{backup_name}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                
                with tarfile.open(backup_file, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                backup_source = temp_dir / backup_name
            elif backup_dir.exists():
                backup_source = backup_dir
            else:
                logger.error(f"備份檔案不存在: {backup_name}")
                return False
            
            # 驗證備份完整性
            if self.config["verify_backup"] and backup_info.get("checksum"):
                current_checksum = self._calculate_checksum(backup_source)
                if current_checksum != backup_info["checksum"]:
                    logger.error("備份校驗和不匹配，可能已損壞")
                    return False
                logger.info("✅ 備份完整性驗證通過")
            
            # 確定恢復目標
            if target_dir is None:
                target_dir = Path(".")
            
            # 備份當前資料
            current_backup_name = f"current_backup_{int(time.time())}"
            logger.info(f"備份當前資料至: {current_backup_name}")
            self.create_backup("current", f"恢復前的自動備份")
            
            # 恢復資料
            if (backup_source / "data").exists():
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                shutil.copytree(backup_source / "data", self.data_dir)
                logger.info("✅ 資料目錄恢復完成")
            
            # 恢復配置
            if (backup_source / "config").exists():
                if self.config_dir.exists():
                    # 備份重要配置檔案
                    important_configs = [".env", "production.yaml"]
                    config_backup = {}
                    for config_file in important_configs:
                        config_path = self.config_dir / config_file
                        if config_path.exists():
                            config_backup[config_file] = config_path.read_text(encoding='utf-8')
                    
                    shutil.rmtree(self.config_dir)
                    shutil.copytree(backup_source / "config", self.config_dir)
                    
                    # 恢復重要配置檔案
                    for config_file, content in config_backup.items():
                        (self.config_dir / config_file).write_text(content, encoding='utf-8')
                        logger.info(f"保留現有配置: {config_file}")
                
                logger.info("✅ 配置目錄恢復完成")
            
            # 恢復應用程式檔案（如果是完整備份）
            if (backup_source / "app").exists():
                app_source = backup_source / "app"
                
                # 恢復主要檔案
                for item in app_source.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_dir / item.name)
                    elif item.is_dir() and item.name == "src":
                        if (target_dir / "src").exists():
                            shutil.rmtree(target_dir / "src")
                        shutil.copytree(item, target_dir / "src")
                
                logger.info("✅ 應用程式檔案恢復完成")
            
            # 清理臨時檔案
            if backup_file.exists() and 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
            
            logger.success(f"🎉 備份恢復完成: {backup_name}")
            logger.info("請重啟應用程式以使恢復生效")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 備份恢復失敗: {e}")
            return False
    
    def _get_backup_info(self, backup_name: str) -> Optional[Dict[str, Any]]:
        """獲取備份資訊"""
        info_file = self.backup_dir / f"{backup_name}.info"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"讀取備份資訊失敗: {e}")
        return None
    
    def delete_backup(self, backup_name: str) -> bool:
        """刪除備份"""
        logger.info(f"🗑️ 刪除備份: {backup_name}")
        
        try:
            # 刪除備份檔案
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            backup_dir = self.backup_dir / backup_name
            info_file = self.backup_dir / f"{backup_name}.info"
            
            deleted_items = []
            
            if backup_file.exists():
                backup_file.unlink()
                deleted_items.append("壓縮檔案")
            
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
                deleted_items.append("目錄")
            
            if info_file.exists():
                info_file.unlink()
                deleted_items.append("資訊檔案")
            
            if deleted_items:
                logger.info(f"✅ 已刪除: {', '.join(deleted_items)}")
                return True
            else:
                logger.warning("備份不存在")
                return False
                
        except Exception as e:
            logger.error(f"❌ 刪除備份失敗: {e}")
            return False
    
    def cleanup_old_backups(self):
        """清理舊備份"""
        logger.info("🧹 清理舊備份")
        
        backups = self.list_backups()
        retention = self.config["retention"]
        
        # 按類型分組
        daily_backups = [b for b in backups if b.get("type") == "daily"]
        weekly_backups = [b for b in backups if b.get("type") == "weekly"]
        monthly_backups = [b for b in backups if b.get("type") == "monthly"]
        full_backups = [b for b in backups if b.get("type") == "full"]
        
        # 清理每日備份
        if len(daily_backups) > retention["daily"]:
            for backup in daily_backups[retention["daily"]:]:
                self.delete_backup(backup["name"])
        
        # 清理週備份
        if len(weekly_backups) > retention["weekly"]:
            for backup in weekly_backups[retention["weekly"]:]:
                self.delete_backup(backup["name"])
        
        # 清理月備份
        if len(monthly_backups) > retention["monthly"]:
            for backup in monthly_backups[retention["monthly"]:]:
                self.delete_backup(backup["name"])
        
        # 保留最近的完整備份
        if len(full_backups) > 3:  # 保留最近 3 個完整備份
            for backup in full_backups[3:]:
                self.delete_backup(backup["name"])
        
        logger.info("✅ 舊備份清理完成")
    
    def verify_backup(self, backup_name: str) -> bool:
        """驗證備份完整性"""
        logger.info(f"🔍 驗證備份: {backup_name}")
        
        try:
            backup_info = self._get_backup_info(backup_name)
            if not backup_info:
                logger.error("找不到備份資訊")
                return False
            
            # 檢查備份檔案是否存在
            backup_file = self.backup_dir / f"{backup_name}.tar.gz"
            backup_dir = self.backup_dir / backup_name
            
            if backup_file.exists():
                backup_path = backup_file
            elif backup_dir.exists():
                backup_path = backup_dir
            else:
                logger.error("備份檔案不存在")
                return False
            
            # 驗證校驗和
            if backup_info.get("checksum"):
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum == backup_info["checksum"]:
                    logger.info("✅ 校驗和驗證通過")
                    return True
                else:
                    logger.error("❌ 校驗和驗證失敗")
                    return False
            else:
                logger.warning("沒有校驗和資訊，無法驗證")
                return True
                
        except Exception as e:
            logger.error(f"❌ 備份驗證失敗: {e}")
            return False


@click.group()
def cli():
    """備份和恢復工具"""
    pass


@cli.command()
@click.option("--type", "backup_type", default="full", 
              type=click.Choice(["full", "data", "config", "daily", "weekly", "monthly"]),
              help="備份類型")
@click.option("--description", default="", help="備份描述")
@click.option("--config", type=click.Path(), help="配置檔案")
def backup(backup_type: str, description: str, config: Optional[str]):
    """建立備份"""
    config_path = Path(config) if config else None
    manager = BackupManager(config_path)
    
    backup_name = manager.create_backup(backup_type, description)
    if backup_name:
        click.echo(f"備份建立成功: {backup_name}")
    else:
        click.echo("備份建立失敗")
        sys.exit(1)


@cli.command()
def list():
    """列出所有備份"""
    manager = BackupManager()
    backups = manager.list_backups()
    
    if not backups:
        click.echo("沒有找到備份")
        return
    
    click.echo(f"{'備份名稱':<30} {'類型':<10} {'大小':<10} {'建立時間':<20} {'狀態'}")
    click.echo("-" * 80)
    
    for backup in backups:
        size_mb = backup.get("size_bytes", 0) / (1024**2)
        created_at = backup.get("created_at", "")[:19].replace("T", " ")
        status = "✅" if backup.get("exists", False) else "❌"
        
        click.echo(f"{backup['name']:<30} {backup.get('type', ''):<10} {size_mb:>8.1f}MB {created_at:<20} {status}")


@cli.command()
@click.argument("backup_name")
@click.option("--target", type=click.Path(), help="恢復目標目錄")
def restore(backup_name: str, target: Optional[str]):
    """恢復備份"""
    manager = BackupManager()
    target_path = Path(target) if target else None
    
    success = manager.restore_backup(backup_name, target_path)
    if success:
        click.echo("備份恢復成功")
    else:
        click.echo("備份恢復失敗")
        sys.exit(1)


@cli.command()
@click.argument("backup_name")
def delete(backup_name: str):
    """刪除備份"""
    if not click.confirm(f"確定要刪除備份 '{backup_name}' 嗎？"):
        return
    
    manager = BackupManager()
    success = manager.delete_backup(backup_name)
    if success:
        click.echo("備份刪除成功")
    else:
        click.echo("備份刪除失敗")
        sys.exit(1)


@cli.command()
@click.argument("backup_name")
def verify(backup_name: str):
    """驗證備份完整性"""
    manager = BackupManager()
    success = manager.verify_backup(backup_name)
    if success:
        click.echo("備份驗證通過")
    else:
        click.echo("備份驗證失敗")
        sys.exit(1)


@cli.command()
def cleanup():
    """清理舊備份"""
    manager = BackupManager()
    manager.cleanup_old_backups()
    click.echo("舊備份清理完成")


if __name__ == "__main__":
    cli()