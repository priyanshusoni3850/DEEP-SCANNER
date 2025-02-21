# Deep Scanner

Deep Scanner is a robust, multi-modal scanning system that leverages deep learning to process images, videos, and audio. The project is structured into separate backend and frontend components to ensure smooth integration between deep model inference and a user-friendly interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Files](#model-files)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Deep Scanner provides a deep learning-based solution for scanning and analyzing multi-modal data. It is designed with a modular architecture that splits the project into:
- **Backend:** Handles API requests and model inference.
- **Frontend:** Offers a seamless user interface for interaction.

---

## Features

- **Multi-Modal Support:** Process images, videos, and audio files using dedicated deep learning models.
- **Easy Integration:** Separate backend and frontend make development and maintenance straightforward.
- **Ngrok Integration:** Securely expose your local backend server using ngrok.
- **Modular Design:** Facilitates updates and additions to the codebase.

---

## Model Files

Before running the application, please download the required models:

- **Image & Video Model:**  
  [Download Image/Video Model](https://drive.google.com/drive/folders/1_-Q0WuoRPdsWTzhmVfxa5hgM9ic3yL7f?usp=sharing)

- **Audio Model:**  
  [Download Audio Model](https://drive.google.com/drive/folders/13McqGsCUpcVjZ9mIZXcetRTlwA50hxBE?usp=sharing)

---

## Installation and Setup

### Prerequisites

- **Node.js** (v14 or higher)
- **Git**
- An active internet connection
- An [ngrok](https://ngrok.com/) account (for secure tunneling)

### Backend Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/priyanshusoni3850/DEEP-SCANNER.git
   cd DEEP-SCANNER/backend
Install Dependencies:

bash
Copy
npm install
Ngrok Setup:

Create an account on ngrok and obtain your auth token.
In the backend code, locate the configuration section or environment variable where the ngrok token is required and insert your token.
Start the Backend Server:

bash
Copy
npm start
Once the backend is running, note the ngrok URL displayed in the console output.
Frontend Setup
Navigate to the Frontend Directory:

bash
Copy
cd ../frontend
Configure the Environment:

Create a .env file in the root directory of the frontend.

Add your backend ngrok URL to the .env file in the following format:

env
Copy
REACT_APP_API_URL=<your-ngrok-url>
Install Frontend Dependencies and Start:

bash
Copy
npm install
npm start
Usage
Access the Frontend:

Open your web browser and navigate to the frontend application.
Upload Files:

Upload an image, video, or audio file to be processed by the deep learning models.
View Results:

The backend processes the file and the results will be displayed on the frontend interface.
Screenshots
Below are placeholders for project images and result screenshots. Replace the image paths with your actual images.

Project Overview:


Result Screenshot 1:


Result Screenshot 2:


You can add up to 2-3 images for both the project and results.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes with clear and descriptive commit messages.
Open a pull request detailing your modifications.
For detailed guidelines, please refer to our CONTRIBUTING.md.

License
This project is licensed under the MIT License.

