#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TShark 字段兼容性检测脚本

功能：
1. 读取本机 tshark 支持的所有字段
2. 读取 config.py 中配置的 TSHARK_FIELDS
3. 检测不兼容的字段
4. 生成兼容性报告和建议

使用方法：
    python check_tshark_fields.py
    
可选参数：
    --update-config    自动更新 config.py，移除不支持的字段
    --export-all       导出所有可用字段到文件
    --recommend        推荐额外有用的字段
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Set, List, Tuple
import argparse


class TSharkFieldChecker:
    """TShark 字段检测器"""
    
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.py"
        self.supported_fields: Set[str] = set()
        self.config_fields: List[str] = []
        
    def check_tshark_available(self) -> bool:
        """检查 tshark 是否可用"""
        try:
            result = subprocess.run(
                ["tshark", "-v"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✓ TShark 已安装: {version_line}")
                return True
            else:
                print("✗ TShark 未正确安装或无法运行")
                return False
        except FileNotFoundError:
            print("✗ 未找到 tshark 命令，请安装 Wireshark")
            return False
        except Exception as e:
            print(f"✗ 检查 tshark 时出错: {e}")
            return False
    
    def get_supported_fields(self) -> Set[str]:
        """获取本机 tshark 支持的所有字段"""
        try:
            result = subprocess.run(
                ["tshark", "-G", "fields"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=30
            )
            
            fields = set()
            for line in result.stdout.split('\n'):
                if line.startswith('F\t'):
                    # 格式: F\t描述\t字段名\t类型\t...
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        field_name = parts[2].strip()
                        if field_name:
                            fields.add(field_name)
            
            self.supported_fields = fields
            print(f"✓ 本机支持 {len(fields)} 个 tshark 字段")
            return fields
            
        except Exception as e:
            print(f"✗ 获取 tshark 字段失败: {e}")
            return set()
    
    def parse_config_fields(self) -> List[str]:
        """从 config.py 中解析 TSHARK_FIELDS"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 TSHARK_FIELDS 列表中的所有字段
            # 匹配引号内的字段名
            pattern = r'"([a-z0-9_\.]+)"'
            matches = re.findall(pattern, content)
            
            # 只保留协议相关字段 (frame, eth, ip, tcp, udp, data 等)
            protocol_fields = [
                m for m in matches 
                if m.startswith(('frame.', 'eth.', 'ip.', 'tcp.', 'udp.', 'data.'))
            ]
            
            # 去重并保持顺序
            seen = set()
            unique_fields = []
            for field in protocol_fields:
                if field not in seen:
                    seen.add(field)
                    unique_fields.append(field)
            
            self.config_fields = unique_fields
            print(f"✓ config.py 配置了 {len(unique_fields)} 个字段")
            return unique_fields
            
        except Exception as e:
            print(f"✗ 读取 config.py 失败: {e}")
            return []
    
    def check_compatibility(self) -> Tuple[List[str], List[str]]:
        """检查字段兼容性
        
        Returns:
            (supported, unsupported) - 支持的字段列表和不支持的字段列表
        """
        supported = []
        unsupported = []
        
        for field in self.config_fields:
            if field in self.supported_fields:
                supported.append(field)
            else:
                unsupported.append(field)
        
        return supported, unsupported
    
    def recommend_fields(self) -> List[str]:
        """推荐一些常用但未配置的字段"""
        recommended_patterns = [
            # HTTP 相关
            r'^http\.(request|response|host|user_agent|cookie)',
            # DNS 相关
            r'^dns\.(qry\.name|resp\.|flags)',
            # TLS/SSL 相关
            r'^tls\.(handshake|record)',
            r'^ssl\.(handshake|record)',
            # ICMP 相关
            r'^icmp\.(type|code)',
            # ARP 相关
            r'^arp\.(src|dst)',
        ]
        
        recommendations = []
        for field in sorted(self.supported_fields):
            if field in self.config_fields:
                continue
            for pattern in recommended_patterns:
                if re.match(pattern, field):
                    recommendations.append(field)
                    break
        
        return recommendations  # 返回所有推荐字段
    
    def export_all_fields(self, output_file: str = "tshark_all_fields.txt"):
        """导出所有可用字段到文件"""
        try:
            output_path = Path(__file__).parent / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                for field in sorted(self.supported_fields):
                    f.write(f"{field}\n")
            print(f"✓ 已导出 {len(self.supported_fields)} 个字段到: {output_path}")
        except Exception as e:
            print(f"✗ 导出字段失败: {e}")
    
    def generate_updated_config(self, supported_fields: List[str]) -> str:
        """生成更新后的字段列表代码"""
        # 按协议分组
        groups = {
            'frame': [],
            'eth': [],
            'ip': [],
            'tcp': [],
            'udp': [],
            'data': [],
        }
        
        for field in supported_fields:
            prefix = field.split('.')[0]
            if prefix in groups:
                groups[prefix].append(field)
        
        # 生成代码
        lines = ["            self.TSHARK_FIELDS = ["]
        
        for protocol in ['frame', 'eth', 'ip', 'tcp', 'udp', 'data']:
            if groups[protocol]:
                lines.append(f"                # {protocol.upper()} 层字段")
                field_strs = [f'"{f}"' for f in groups[protocol]]
                lines.append("                " + ", ".join(field_strs) + ",")
                lines.append("")
        
        # 移除最后的空行和逗号
        if lines[-1] == "":
            lines.pop()
        lines[-1] = lines[-1].rstrip(',')
        lines.append("            ]")
        
        return '\n'.join(lines)
    
    def update_config_file(self, new_fields_code: str):
        """更新 config.py 文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找并替换 TSHARK_FIELDS 部分
            pattern = r'(self\.TSHARK_FIELDS = \[).*?(\])'
            
            # 使用 DOTALL 模式匹配多行
            new_content = re.sub(
                pattern,
                new_fields_code.replace('            self.TSHARK_FIELDS = [', r'\1').replace('            ]', r'\2'),
                content,
                flags=re.DOTALL
            )
            
            if new_content != content:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ 已更新 config.py")
            else:
                print("⚠ config.py 未发生变化")
                
        except Exception as e:
            print(f"✗ 更新 config.py 失败: {e}")
    
    def print_report(self, supported: List[str], unsupported: List[str], recommendations: List[str] = None):
        """打印检测报告"""
        print("\n" + "="*70)
        print("TShark 字段兼容性检测报告")
        print("="*70)
        
        print(f"\n 统计信息:")
        print(f"   配置字段总数: {len(self.config_fields)}")
        print(f"   本机支持数量: {len(supported)} ({len(supported)/len(self.config_fields)*100:.1f}%)")
        print(f"   不支持数量:   {len(unsupported)} ({len(unsupported)/len(self.config_fields)*100:.1f}%)")
        
        if unsupported:
            print(f"\n  以下 {len(unsupported)} 个字段在本机不支持:")
            for field in unsupported:
                print(f"   ✗ {field}")
            print("\n  建议：从 config.py 中移除这些字段，或使用 --update-config 自动更新")
        else:
            print("\n  所有配置的字段在本机都支持！")
        
        if recommendations:
            print(f"\n  推荐添加的常用字段 ( {len(recommendations)} 个):")
            for field in recommendations:
                print(f"   + {field}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='TShark 字段兼容性检测工具')
    parser.add_argument('--update-config', action='store_true', 
                        help='自动更新 config.py，移除不支持的字段')
    parser.add_argument('--export-all', action='store_true',
                        help='导出所有可用字段到文件')
    parser.add_argument('--recommend', action='store_true',
                        help='推荐额外有用的字段')
    
    args = parser.parse_args()
    
    checker = TSharkFieldChecker()
    
    # 1. 检查 tshark 是否可用
    if not checker.check_tshark_available():
        sys.exit(1)
    
    # 2. 获取支持的字段
    if not checker.get_supported_fields():
        sys.exit(1)
    
    # 3. 解析配置文件
    if not checker.parse_config_fields():
        sys.exit(1)
    
    # 4. 导出所有字段（如果需要）
    if args.export_all:
        checker.export_all_fields()
    
    # 5. 检查兼容性
    supported, unsupported = checker.check_compatibility()
    
    # 6. 获取推荐字段（如果需要）
    recommendations = None
    if args.recommend:
        recommendations = checker.recommend_fields()
    
    # 7. 打印报告
    checker.print_report(supported, unsupported, recommendations)
    
    # 8. 更新配置文件（如果需要）
    if args.update_config and unsupported:
        print("\n正在更新 config.py...")
        new_code = checker.generate_updated_config(supported)
        print("\n生成的新字段配置:")
        print(new_code)
        
        confirm = input("\n确认更新 config.py? (y/N): ")
        if confirm.lower() == 'y':
            checker.update_config_file(new_code)
        else:
            print("已取消更新")
    
    # 9. 返回退出码
    sys.exit(0 if not unsupported else 1)


if __name__ == "__main__":
    main()
