import requests
from typing import TypedDict, Optional
from langgraph.graph import StateGraph

# --- State schema ---
class WeatherState(TypedDict, total=False):
    user_input: str
    intent: str
    city: Optional[str]
    weather: str

# --- CONFIG ---
OPENWEATHERMAP_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # TODO: Insert your real API key here
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"  # TODO: Insert your real API key here

# --- DeepSeek LLM mock (replace with real call) ---
def analyze_intent_with_deepseek(user_input):
    """
    Mock function to simulate DeepSeek LLM intent extraction.
    Replace with real DeepSeek API call.
    """
    # For demo: extract city if user says 'weather in <city>'
    import re
    match = re.search(r"weather in ([a-zA-Z ]+)", user_input, re.IGNORECASE)
    if match:
        return {"intent": "get_weather", "city": match.group(1).strip()}
    return {"intent": "unknown", "city": None}

# --- OpenWeatherMap API call ---
def get_weather_for_city(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
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
    compiled = graph.compile()
    compiled.invoke({})

if __name__ == "__main__":
    main()