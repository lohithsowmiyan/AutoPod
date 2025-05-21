import os
from typing import Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from mad import MAD

sarah = """
You are Sarah, an experienced and thoughtful individual. You focus on the quality of the content and prefer concise, well-structured information. You have a strong appreciation for traditional topics such as sports, history, and established cultural themes.
"""
john = """
You are John, a young and energetic voice who brings fresh perspectives to the conversation. You enjoy presenting content in a lively, engaging way and aren’t afraid to explore controversial or provocative topics like crime, relationships, and social issues.
"""

def load_prompt_template() -> PromptTemplate:
    news_recitation_prompt = PromptTemplate(
    input_variables=["persona", "news_content", "duration"],
    template="""
    {persona} 
    Please use the following content clearly and concisely to create an engaging, well-structured podcast episode. Ensure the script includes multiple speakers to create a dynamic and immersive listening experience.

    To indicate who is speaking, use [S1], [S2], [S3], etc., before each line, where S stands for Speaker and N is the speaker number.

    Please use the following expressions to enhance realism and tone but do not use expressions other than the ones listed below:
    (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles).

    Do not use any other special characters or symbols.

    Tone: Conversational, emotionally resonant, and informative.
    Style: A balanced mix of storytelling and factual discussion to maintain listener interest throughout.

    Structure:
    1. Opening Hook (30s–1 min): Start with a striking or emotionally resonant moment to immediately capture attention.
    2. Background & Context (1–2 mins): Explain the key people, events, or issues involved in a way that’s clear and accessible.
    3. Main Developments (2–3 mins): Discuss what happened, unfolding the story in a compelling sequence with important quotes or soundbites.
    4. Personal Voices / Testimonies (1–2 mins): Include voices of those directly involved, expert opinions, or moving accounts.
    5. Closing Thoughts / What’s Next (30s–1 min): Wrap up with the current state of things and what listeners should watch out for.

    Maintain clarity, pacing, and vivid narration.

    Content
    ---
    {news_content}
    ---

    Generate a news script optimized for a spoken news segment of approximately {duration} minutes.
    """
    )

    return news_recitation_prompt

def summarize_contents(content: Dict[str, str]) -> Dict[str, str]:
    prompt_template = load_prompt_template()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.8
    )
    initial_respones = []
    for i_name in [sarah,john]:
        prompt = prompt_template.format(persona=i_name,news_content=content[:10000],duration= 5)  # Trim long input
        response = llm.invoke(prompt)
        initial_respones.append(response.content)
    mad_agents = MAD(initial_respones[0], initial_respones[1])
    return mad_agents.debate(llm)
        

