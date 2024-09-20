import os

# Set environment variables and install llama_cpp_python for gpu
os.system('CMAKE_ARGS="-DLLAMA=on" FORCE_CMAKE=1 pip install llama-cpp-python')

# Run pip install for requirements.txt
os.system("pip install -r requirements.txt")