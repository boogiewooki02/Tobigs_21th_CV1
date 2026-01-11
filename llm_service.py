import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SolarLLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SOLAR_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.risk_desc = ["안전(Original)", "주의(Low)", "경고(Medium)", "위험(High)"]

    def generate_report(self, vit_results):
        prompt = f"""
        당신은 이미지 위변조 분석 전문가입니다. 아래 데이터를 바탕으로 리포트를 작성하세요.
        - 등급: {self.risk_desc[vit_results['label']]}
        - SSIM: {vit_results['ssim']:.4f}
        - Strength: {vit_results['strength']:.2f}
        
        전문적이면서 일반 사용자도 이해하기 쉽게 설명해 주세요.
        """
        
        response = self.client.chat.completions.create(
            model="solar-pro2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class GPTLLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.risk_desc = ["안전(Original)", "주의(Low)", "경고(Medium)", "위험(High)"]

    def generate_report(self, vit_results):
        prompt = f"""
        당신은 이미지 위변조 분석 전문가입니다. 아래 데이터를 바탕으로 리포트를 작성하세요.
        - 등급: {self.risk_desc[vit_results['label']]}
        - SSIM: {vit_results['ssim']:.4f}
        - 변형 강도(Strength): {vit_results['strength']:.2f}
        
        전문적이면서 일반 사용자도 이해하기 쉽게 설명해 주세요.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content