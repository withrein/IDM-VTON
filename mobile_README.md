# IDM-VTON Mobile Integration

This project provides tools to integrate the IDM-VTON virtual try-on system with mobile applications.

![IDM-VTON Demo](assets/teaser.png)

## Features

- Virtual try-on for clothing items on mobile devices
- Client-server architecture using Gradio as the backend API
- Support for iOS and Android through React Native, Flutter, or native development
- Easy to integrate with existing e-commerce apps

## Getting Started

### 1. Server Setup

First, you need to set up the Gradio server that will handle the try-on processing:

```bash
# Clone the repository
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

# Install dependencies
conda env create -f environment.yaml
conda activate idm

# Download model checkpoints as instructed in the main README

# Start the mobile-optimized Gradio server
python gradio_demo/mobile_app.py
```

The server will start and display a URL like: `Running on local URL: http://0.0.0.0:7860`

### 2. Mobile App Integration

Choose your preferred mobile development approach:

- **React Native**: Use the example code in `mobile_integration.md`
- **Flutter**: Check the Flutter example in `mobile_integration.md`
- **Native iOS/Android**: See the Swift and Kotlin examples

### 3. Testing

Test your integration by:

1. Taking a photo or selecting an image of a person
2. Selecting a garment image
3. Providing a description of the garment
4. Clicking "Try It On" to see the result

## Architecture

The system uses a client-server architecture:

1. The mobile app captures/selects images and sends them to the server
2. The server processes the images using the IDM-VTON model
3. The server returns the try-on result to the mobile app
4. The mobile app displays the result

## Deployment Options

### Development/Testing

- Run the server on a local machine with a GPU
- Connect mobile devices on the same network for testing

### Production

- Deploy the server on a cloud platform (AWS, GCP, Azure)
- Set up proper API authentication and scaling
- Connect your production mobile app to the cloud API

## Mobile UI Components

The mobile interface provides:

- Camera/photo selection for the user's image
- Garment image selection
- Text description input
- Try-on button and result display
- Progress indicator

## Performance Optimization

For better performance:

- Resize images on the mobile device before sending to the server
- Use a queue system for handling multiple requests
- Implement caching for frequent garment try-ons
- Consider server-side scaling for production deployments

## Further Documentation

For detailed documentation:

- See `mobile_integration.md` for code examples and API details
- Check the main `README.md` for information about the core model
- Visit the [project website](https://idm-vton.github.io) for more information

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Citation

```
@article{choi2024improving,
  title={Improving Diffusion Models for Authentic Virtual Try-on in the Wild},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
``` 