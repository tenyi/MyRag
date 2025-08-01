#!/usr/bin/env python3
"""
測試結果通知器

根據測試結果和品質閘門狀態發送通知到各種渠道。
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import yaml


class TestNotifier:
    """測試結果通知器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化通知器
        
        Args:
            config_file: 通知配置檔案路徑
        """
        self.config = self._load_config(config_file)
        self.test_results = {}
        self.quality_gate_results = {}
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """載入通知配置"""
        default_config = {
            "notifications": {
                "enabled": True,
                "conditions": ["on_failure"],
                "channels": {
                    "slack": {"enabled": False},
                    "email": {"enabled": False},
                    "github": {"enabled": True, "add_pr_comment": True}
                }
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 深度合併配置
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """深度合併字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def load_test_results(self, results_file: str):
        """載入測試結果"""
        if Path(results_file).exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                self.test_results = json.load(f)
    
    def load_quality_gate_results(self, results_file: str):
        """載入品質閘門結果"""
        if Path(results_file).exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                self.quality_gate_results = json.load(f)
    
    def should_notify(self) -> bool:
        """判斷是否應該發送通知"""
        if not self.config["notifications"]["enabled"]:
            return False
        
        conditions = self.config["notifications"]["conditions"]
        overall_status = self.quality_gate_results.get("overall_status", "UNKNOWN")
        
        # 檢查通知條件
        if "on_failure" in conditions and overall_status == "FAIL":
            return True
        if "on_success" in conditions and overall_status == "PASS":
            return True
        if "on_quality_regression" in conditions:
            # 這裡可以添加品質回歸檢測邏輯
            pass
        
        return False
    
    def generate_notification_content(self) -> Dict[str, str]:
        """生成通知內容"""
        overall_status = self.quality_gate_results.get("overall_status", "UNKNOWN")
        checks = self.quality_gate_results.get("checks", {})
        violations = self.quality_gate_results.get("violations", [])
        metrics = self.quality_gate_results.get("metrics", {})
        
        # 狀態圖示
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "UNKNOWN": "❓"
        }.get(overall_status, "❓")
        
        # 生成標題
        title = f"{status_icon} Chinese GraphRAG 測試結果: {overall_status}"
        
        # 生成摘要
        summary_parts = []
        if "overall_pass_rate" in metrics:
            summary_parts.append(f"整體通過率: {metrics['overall_pass_rate']:.1f}%")
        if "code_coverage" in metrics:
            summary_parts.append(f"程式碼覆蓋率: {metrics['code_coverage']:.1f}%")
        
        summary = " | ".join(summary_parts) if summary_parts else "無可用指標"
        
        # 生成詳細內容
        details = []
        details.append(f"## 📊 測試結果摘要")
        details.append(f"- **整體狀態**: {overall_status}")
        details.append(f"- **檢查時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        details.append(f"- **Git 分支**: {os.getenv('GITHUB_REF_NAME', 'unknown')}")
        details.append(f"- **Git 提交**: {os.getenv('GITHUB_SHA', 'unknown')[:8]}")
        details.append("")
        
        # 各項檢查結果
        if checks:
            details.append("## 🔍 檢查項目")
            for check_name, check_data in checks.items():
                status = check_data.get("status", "UNKNOWN")
                message = check_data.get("message", "")
                check_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "MISSING": "⚠️",
                    "ERROR": "🔥"
                }.get(status, "❓")
                
                details.append(f"- {check_icon} **{check_name}**: {message}")
                
                if "value" in check_data and "threshold" in check_data:
                    details.append(f"  - 數值: {check_data['value']:.1f}, 閾值: {check_data['threshold']}")
            details.append("")
        
        # 違規項目
        if violations:
            details.append("## ❌ 發現的問題")
            for violation in violations:
                details.append(f"- {violation}")
            details.append("")
        
        # 關鍵指標
        if metrics:
            details.append("## 📈 關鍵指標")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    details.append(f"- **{metric}**: {value:.1f}%")
                else:
                    details.append(f"- **{metric}**: {value}")
            details.append("")
        
        # 添加操作建議
        if overall_status == "FAIL":
            details.append("## 🔧 建議操作")
            details.append("1. 檢查失敗的測試案例")
            details.append("2. 分析程式碼覆蓋率報告")
            details.append("3. 修復發現的問題")
            details.append("4. 重新執行測試")
        
        return {
            "title": title,
            "summary": summary,
            "details": "\n".join(details),
            "status": overall_status
        }
    
    def send_slack_notification(self, content: Dict[str, str]) -> bool:
        """發送 Slack 通知"""
        slack_config = self.config["notifications"]["channels"]["slack"]
        
        if not slack_config.get("enabled", False):
            return True
        
        webhook_url = os.getenv("SLACK_WEBHOOK_URL") or slack_config.get("webhook_url")
        if not webhook_url:
            print("警告: 未設定 Slack webhook URL")
            return False
        
        # 構建 Slack 訊息
        color = "good" if content["status"] == "PASS" else "danger"
        
        payload = {
            "channel": slack_config.get("channel", "#ci-cd"),
            "username": "Chinese GraphRAG CI",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": content["title"],
                    "text": content["summary"],
                    "fields": [
                        {
                            "title": "詳細資訊",
                            "value": content["details"][:1000] + "..." if len(content["details"]) > 1000 else content["details"],
                            "short": False
                        }
                    ],
                    "footer": "Chinese GraphRAG CI/CD",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print("✅ Slack 通知發送成功")
            return True
        except Exception as e:
            print(f"❌ Slack 通知發送失敗: {e}")
            return False
    
    def send_email_notification(self, content: Dict[str, str]) -> bool:
        """發送郵件通知"""
        email_config = self.config["notifications"]["channels"]["email"]
        
        if not email_config.get("enabled", False):
            return True
        
        smtp_server = os.getenv("SMTP_SERVER") or email_config.get("smtp_server")
        smtp_port = email_config.get("smtp_port", 587)
        smtp_user = os.getenv("SMTP_USER") or email_config.get("smtp_user")
        smtp_password = os.getenv("SMTP_PASSWORD") or email_config.get("smtp_password")
        recipients = email_config.get("recipients", [])
        
        if not all([smtp_server, smtp_user, smtp_password, recipients]):
            print("警告: 郵件配置不完整")
            return False
        
        try:
            # 建立郵件
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = content["title"]
            
            # 郵件內容
            body = f"""
{content['summary']}

{content['details']}

---
此郵件由 Chinese GraphRAG CI/CD 系統自動發送
時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 發送郵件
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()
            
            print("✅ 郵件通知發送成功")
            return True
            
        except Exception as e:
            print(f"❌ 郵件通知發送失敗: {e}")
            return False
    
    def send_github_notification(self, content: Dict[str, str]) -> bool:
        """發送 GitHub 通知"""
        github_config = self.config["notifications"]["channels"]["github"]
        
        if not github_config.get("enabled", False):
            return True
        
        # 如果是 PR，添加評論
        if github_config.get("add_pr_comment", False) and os.getenv("GITHUB_EVENT_NAME") == "pull_request":
            return self._add_pr_comment(content)
        
        # 如果失敗且配置了建立 issue，則建立 issue
        if (content["status"] == "FAIL" and 
            github_config.get("create_issue_on_failure", False)):
            return self._create_github_issue(content)
        
        return True
    
    def _add_pr_comment(self, content: Dict[str, str]) -> bool:
        """在 PR 中添加評論"""
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository = os.getenv("GITHUB_REPOSITORY")
        pr_number = os.getenv("GITHUB_PR_NUMBER")
        
        if not all([github_token, github_repository, pr_number]):
            print("警告: GitHub PR 評論所需的環境變數不完整")
            return False
        
        url = f"https://api.github.com/repos/{github_repository}/issues/{pr_number}/comments"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        comment_body = f"""## {content['title']}

{content['summary']}

<details>
<summary>詳細資訊</summary>

{content['details']}

</details>

---
*此評論由 Chinese GraphRAG CI/CD 系統自動生成*
"""
        
        payload = {"body": comment_body}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print("✅ GitHub PR 評論添加成功")
            return True
        except Exception as e:
            print(f"❌ GitHub PR 評論添加失敗: {e}")
            return False
    
    def _create_github_issue(self, content: Dict[str, str]) -> bool:
        """建立 GitHub issue"""
        github_token = os.getenv("GITHUB_TOKEN")
        github_repository = os.getenv("GITHUB_REPOSITORY")
        
        if not all([github_token, github_repository]):
            print("警告: GitHub issue 建立所需的環境變數不完整")
            return False
        
        url = f"https://api.github.com/repos/{github_repository}/issues"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        issue_title = f"CI/CD 品質閘門失敗 - {datetime.now().strftime('%Y-%m-%d')}"
        issue_body = f"""## {content['title']}

{content['details']}

### 環境資訊
- **分支**: {os.getenv('GITHUB_REF_NAME', 'unknown')}
- **提交**: {os.getenv('GITHUB_SHA', 'unknown')[:8]}
- **工作流程**: {os.getenv('GITHUB_WORKFLOW', 'unknown')}
- **執行 ID**: {os.getenv('GITHUB_RUN_ID', 'unknown')}

---
*此 issue 由 Chinese GraphRAG CI/CD 系統自動建立*
"""
        
        payload = {
            "title": issue_title,
            "body": issue_body,
            "labels": ["ci/cd", "quality-gate", "bug"]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print("✅ GitHub issue 建立成功")
            return True
        except Exception as e:
            print(f"❌ GitHub issue 建立失敗: {e}")
            return False
    
    def send_notifications(self) -> bool:
        """發送所有通知"""
        if not self.should_notify():
            print("ℹ️ 不符合通知條件，跳過通知發送")
            return True
        
        print("📢 開始發送測試結果通知...")
        
        content = self.generate_notification_content()
        
        success = True
        
        # 發送 Slack 通知
        if not self.send_slack_notification(content):
            success = False
        
        # 發送郵件通知
        if not self.send_email_notification(content):
            success = False
        
        # 發送 GitHub 通知
        if not self.send_github_notification(content):
            success = False
        
        if success:
            print("✅ 所有通知發送完成")
        else:
            print("⚠️ 部分通知發送失敗")
        
        return success


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description="測試結果通知器")
    parser.add_argument("--config", help="通知配置檔案")
    parser.add_argument("--test-results", help="測試結果檔案")
    parser.add_argument("--quality-gate-results", default="quality-gate-results.json", help="品質閘門結果檔案")
    
    args = parser.parse_args()
    
    # 建立通知器
    notifier = TestNotifier(args.config)
    
    # 載入結果
    if args.test_results:
        notifier.load_test_results(args.test_results)
    
    notifier.load_quality_gate_results(args.quality_gate_results)
    
    # 發送通知
    success = notifier.send_notifications()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()