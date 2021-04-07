mkdir -p ~/.streamlit/

echo"\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\m\
\n\
">~/.streamlit/config.toml
