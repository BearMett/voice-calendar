"""
설정 파일 모듈

이 모듈은 애플리케이션의 설정을 관리합니다.
"""

import os
import json
import yaml


class Config:
    """설정 관리 클래스"""

    def __init__(self, config_path="config.yaml"):
        """
        Config 초기화

        Args:
            config_path (str): 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """설정 파일 로드"""
        if not os.path.exists(self.config_path):
            # 기본 설정 생성
            default_config = {
                "calendar": {
                    "type": "google",
                    "credentials_path": "gcloud/credentials.json",
                    "token_path": "token.json",
                },
                "speech": {
                    "model_name": "openai/whisper-large-v3",
                    "language": "korean",
                    "cache_dir": "models",
                    "silence_threshold": 1000,
                    "silence_duration": 2.0,
                    "max_duration": 60,
                },
                "llm": {
                    "model_type": "huggingface",
                    "model_name": "beomi/KoAlpaca-Polyglot-5.8B",
                    "cache_dir": "models",
                },
            }

            # 설정 파일 저장
            self._save_config(default_config)
            return default_config

        # 파일 확장자에 따라 로드 방식 결정
        ext = os.path.splitext(self.config_path)[1].lower()

        try:
            if ext == ".json":
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif ext in [".yaml", ".yml"]:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            else:
                print(f"지원하지 않는 설정 파일 형식: {ext}")
                return self._create_default_config()
        except Exception as e:
            print(f"설정 파일 로드 오류: {e}")
            return self._create_default_config()

    def _create_default_config(self):
        """기본 설정 생성"""
        default_config = {
            "calendar": {
                "type": "google",
                "credentials_path": "gcloud/credentials.json",
                "token_path": "token.json",
            },
            "speech": {
                "model_name": "openai/whisper-large-v3",
                "language": "korean",
                "cache_dir": "models",
                "silence_threshold": 1000,
                "silence_duration": 2.0,
                "max_duration": 60,
            },
            "llm": {
                "model_type": "huggingface",
                "model_name": "beomi/KoAlpaca-Polyglot-5.8B",
                "cache_dir": "models",
            },
        }

        return default_config

    def _save_config(self, config=None):
        """
        설정 파일 저장

        Args:
            config (dict): 저장할 설정 (None인 경우 현재 설정 사용)
        """
        if config is None:
            config = self.config

        # 파일 확장자에 따라 저장 방식 결정
        ext = os.path.splitext(self.config_path)[1].lower()

        try:
            if ext == ".json":
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            elif ext in [".yaml", ".yml"]:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                # 기본적으로 YAML 형식으로 저장
                with open(self.config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"설정 파일 저장 오류: {e}")

    def get(self, section, key=None, default=None):
        """
        설정 값 가져오기

        Args:
            section (str): 설정 섹션
            key (str): 설정 키 (None인 경우 섹션 전체 반환)
            default: 기본값

        Returns:
            설정 값
        """
        if section not in self.config:
            return default

        if key is None:
            return self.config[section]

        return self.config[section].get(key, default)

    def set(self, section, key, value):
        """
        설정 값 설정

        Args:
            section (str): 설정 섹션
            key (str): 설정 키
            value: 설정 값
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value
        self._save_config()

    def update(self, section, values):
        """
        설정 섹션 업데이트

        Args:
            section (str): 설정 섹션
            values (dict): 업데이트할 값들
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section].update(values)
        self._save_config()
