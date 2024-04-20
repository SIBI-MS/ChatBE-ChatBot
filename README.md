
# ChatBE

ChatBE is an open-source chatbot that connects to a MySQL database, allowing users to ask questions based on the database content. It is built using the Langchain framework, Langsmith for monitoring, and the Llama-2-7b-chat-hf model. The user interface is created with Streamlit.

## Features

- Connects to a MySQL database to retrieve and interact with data.
- Uses the Llama-2-7b-chat-hf model for natural language understanding and generation.
- Monitored by Langsmith for performance and reliability.
- Provides a user-friendly interface powered by Streamlit.

## Requirements

To run ChatBE, you'll need the following dependencies installed:

- Python 3.x
- Langchain
- Langsmith
- Streamlit
- MySQL client library

You can install the required Python packages using pip:

```
pip install langchain langsmith streamlit mysql-connector-python
```

## Usage

1. Clone the ChatBE repository:

```
git clone https://github.com/yourusername/ChatBE.git
cd ChatBE
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Configure the MySQL database connection by updating the `config.ini` file with your database credentials.

4. Run the Streamlit app:

```
streamlit run app.py
```

5. Access the chatbot interface in your web browser by navigating to the provided URL.

## Contributing

Contributions to ChatBE are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and ensure that the code passes all tests.
3. Submit a pull request with a clear description of your changes.


## Acknowledgements

- The Langchain framework for providing a powerful platform for building conversational AI.
- Langsmith for monitoring and ensuring the reliability of ChatBE.
- The Llama-2-7b-chat-hf model for natural language processing capabilities.
- Streamlit for creating a user-friendly interface for ChatBE.

## Contact

For questions, feedback, or support, please contact [your@email.com](mailto:sibims00@email.com).
