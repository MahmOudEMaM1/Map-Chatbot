# Place Information Chatbot

This project integrates Hugging Face's `transformers` library with HERE API to create a chatbot that generates detailed descriptions of locations and provides map links to their coordinates.

## Key Components:
- **Text Generation**: The model uses Meta’s `Llama-3.2-3B-Instruct` to generate descriptive text about a given location.
- **HERE API Integration**: The code fetches geographical coordinates (latitude and longitude) for the specified location and generates a link to the map.
- **Quantization**: The model is optimized using 4-bit quantization with `BitsAndBytesConfig` for efficient performance.

## Functionality:
1. **`generate_place_info(place_name)`**: Generates a description of the specified place using a text-generation pipeline.
2. **`get_here_coordinates(place_name)`**: Fetches the coordinates (latitude and longitude) of a place using the `HERE geocoding API`.
3. **`get_here_map_link(lat_lng)`**: Generates a map link for the location's coordinates.
4. **`chatbot_flow(place_name)`**: Combines the place description and coordinates into a user-friendly response with a map link.

## Colab Notebook
To explore the code and test the chatbot yourself, you can use the following [Colab notebook link](https://colab.research.google.com/drive/146ec09CkM0uNS8e51MLmV0SGuYfSonaD#scrollTo=UxLxzDyYTBkO).

### Example usage:

```python
place_name = "name of place here"
print(chatbot_flow(place_name))

