import os
from typing import Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



def load_prompt_template() -> PromptTemplate:
    news_recitation_prompt = PromptTemplate(
    input_variables=["news_content", "duration"],
    template="""
    You are a professional news anchor preparing a live news segment.
    Use the following news content to craft an engaging and coherent news report that lasts approximately {duration} minutes.

    Tone: Authoritative, emotionally aware, and respectful.
    Style: A blend of journalistic reporting and narrative storytelling to maintain audience engagement.

    Structure:
    1. Headline Hook (30s–1 min): Begin with a compelling summary of the most impactful or dramatic part of the news.
    2. Context and Background (1–2 mins): Provide key background on people, events, or issues involved.
    3. Main Developments (2–3 mins): Describe the sequence of major events or updates. Include relevant quotes or details.
    4. Key Witnesses/Testimonies/Details (1–2 mins): Highlight compelling human angles or official statements.
    5. Current Status & What’s Next (30s–1 min): Explain what is happening now and what’s expected next.

    Keep the language professional yet conversational — suitable for TV or radio broadcast.

    Input News:
    \"\"\"
    {news_content}
    \"\"\"

    Generate a news script optimized for a spoken news segment of approximately {duration} minutes.
    """
    )

    return news_recitation_prompt

def summarize_contents(content: Dict[str, str]) -> Dict[str, str]:
    prompt_template = load_prompt_template()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    prompt = prompt_template.format(news_content=content[:10000],duration= 5)  # Trim long input
    response = llm.invoke(prompt)
    summaries = response.content
    return summaries
