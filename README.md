# ADK learning

Agents are basically an implementation of LLMs, to make them autonomous and to perform actions
on their own without human intervention
they are basically a comination of
1. model
2. prompt
3. tools

## Current Project Status

Ollama currently does not posses an adk agent in the kit
but this is not a problem, the agent simply expects our model struct
to have two functions

- `Name` -> String containing the name of the model used (qwen2.5-coder:7b)
- `GenerateContent` -> yield which is basically an iterative return type allowing the typing style responses llms give to reduce the lag between q/a

At the moment it works by using a private network connection to my Computer with a good GPU
and basically makes an api call to the ollama endpoint and parses it to the format expected by the adk
library. It is a basic chat wrapper with no tools at the moment but its a starting point to more interesting
stuff

