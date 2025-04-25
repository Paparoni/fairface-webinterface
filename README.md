# FairFace Web Interface

**An AI-powered web application for demographic analysis using the FairFace model.**

This project provides a simple and intuitive web interface for the FairFace facial recognition model. Users can upload images and receive demographic predictions, including age, gender, and ethnicity. Powered by artificial intelligence, the app focuses on inclusivity and accuracy across diverse facial features.

## Features

- 🌐 Web-based interface for easy image upload and analysis  
- 🤖 AI-driven demographic predictions  
- 📊 Outputs include age, gender, and ethnicity  
- 🎯 Built for clarity, speed, and accessibility  

## Project Structure

- `app.py`: Flask web server that runs the application  
- `predict.py`: Handles image processing and model predictions  
- `fair_face_models/`: Contains the FairFace model files  
- `dlib_models/`: Contains required dlib model files for face detection  
- `templates/` and `static/`: Frontend HTML templates and assets  

## Acknowledgements

- **[FairFace](https://github.com/joojs/fairface)** – for the open-source facial recognition model focused on diversity and inclusion  
- **[dlib](http://dlib.net/)** – for face detection  
- **[OpenCV](https://opencv.org/)** – for image handling and processing  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
