import asyncio
from google.genai import types
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-live-001"

# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}



config = types.LiveConnectConfig(
    response_modalities=["TEXT"],
    tools=[
        types.Tool(
            function_declarations=[set_light_values_declaration]
        )
    ]
)

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        await session.send_client_content(
            turns={
                "role": "user",
                "parts": [{"text": "Please use the set_light_values function to turn the lights down to a romantic level with warm temperature"}],
            },
            turn_complete=True,
        )

        try:
            async for response in session.receive():
                print("\n[RESPONSE]")
                if response.text:
                    print("Text:", response.text)
                print("[TOOL CALL]", response.tool_call)
                if response.tool_call:
                    print("[TOOL CALL DETAILS]", response.tool_call.function_calls)
                    # Execute the function call
                    for function_call in response.tool_call.function_calls:
                        if function_call.name == "set_light_values":
                            result = set_light_values(**function_call.args)
                            print("[FUNCTION RESULT]", result)
                            # Exit the loop after executing the tool call
                            break
                # Exit the outer loop after processing the tool call
                if response.tool_call:
                    break
        except Exception as e:
            print(f"Error during response processing: {e}")

if __name__ == "__main__":
    asyncio.run(main())