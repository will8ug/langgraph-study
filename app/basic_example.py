import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
import os
import json
import traceback
import re
from dotenv import load_dotenv

# --- State schema ---
class WeatherState(TypedDict, total=False):
    user_input: str
    intent: str
    city: Optional[str]
    weather: str

def extract_json_from_markdown(text):
    """
    Extract JSON from markdown code blocks like ```json ... ```
    """
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # If no markdown blocks, try to find JSON directly
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def analyze_intent_with_deepseek(user_input):
    """
    Uses langchain-deepseek to extract intent and city from user input.
    """
    try:
        chat = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.0
        )
        
        # Create the system and user messages
        system_message = SystemMessage(content=(
            "You are an intent extraction assistant. "
            "Given a user question about the weather, extract the intent (get_weather or unknown) "
            "and the city name if present. "
            "Respond in JSON like: {\"intent\": \"get_weather\", \"city\": \"Paris\"} or {\"intent\": \"unknown\", \"city\": null}."
        ))
        user_message = HumanMessage(content=user_input)
        
        # Get the response from DeepSeek
        response = chat.invoke([system_message, user_message])
        content = response.content
        
        print(f"Debug - Raw DeepSeek response: {content}")
        
        # Extract JSON from markdown if present
        json_content = extract_json_from_markdown(content)
        print(f"Debug - Extracted JSON: {json_content}")
        
        # Parse the JSON response
        parsed = json.loads(json_content)
        return {"intent": parsed.get("intent", "unknown"), "city": parsed.get("city")}
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        print("Full error stack:")
        traceback.print_exc()
        return {"intent": "unknown", "city": None}

# --- OpenWeatherMap API call ---
def get_weather_for_city(city):
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        desc = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Weather in {city}: {desc}, {temp}°C"
    else:
        return f"Could not get weather for {city}."

# --- LangGraph nodes ---
def input_node(state: WeatherState) -> WeatherState:
    user_input = input("Ask about the weather (e.g., 'What's the weather in Paris?'): ")
    state['user_input'] = user_input
    return state

def analyze_node(state: WeatherState) -> WeatherState:
    result = analyze_intent_with_deepseek(state['user_input'])
    state.update(result)
    return state

def weather_node(state: WeatherState) -> WeatherState:
    if state.get('intent') == 'get_weather' and state.get('city'):
        state['weather'] = get_weather_for_city(state['city'])
    else:
        state['weather'] = "Sorry, I couldn't understand your request."
    return state

def output_node(state: WeatherState) -> WeatherState:
    print(state.get('weather', 'No weather info.'))
    return state

# --- Build the graph ---
graph = StateGraph(WeatherState)
graph.add_node("input", input_node)
graph.add_node("analyze", analyze_node)
graph.add_node("weather", weather_node)
graph.add_node("output", output_node)
graph.add_edge("input", "analyze")
graph.add_edge("analyze", "weather")
graph.add_edge("weather", "output")
graph.set_entry_point("input")

def main():
    load_dotenv()
    
    compiled = graph.compile()
    compiled.invoke({})

if __name__ == "__main__":
    main()