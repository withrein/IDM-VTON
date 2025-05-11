`# Integrating IDM-VTON with Mobile Applications

This document provides instructions for integrating the IDM-VTON virtual try-on system with mobile applications using the Gradio API.

## Overview

IDM-VTON is a diffusion-based virtual try-on system that allows users to visualize how clothing items would look on them. The mobile integration works by:

1. Running a Gradio server that hosts the try-on model
2. Connecting to this server from a mobile app using the Gradio client API or direct HTTP requests

## Setup Options

### Option 1: Cloud Hosted Gradio (Recommended for Production)

1. Deploy the Gradio interface on a cloud server (AWS, GCP, Azure)
2. Expose the API endpoint securely
3. Connect your mobile app to this endpoint

### Option 2: Local Network for Testing

1. Run the Gradio server on a local machine with GPU
2. Connect mobile devices on the same network to the server
3. Use for development and testing

## Server Setup

### Requirements

Ensure your server has:
- Python 3.10+
- PyTorch with CUDA support
- All dependencies listed in `environment.yaml`
- Sufficient GPU memory (8GB+ recommended)

### Launching the Server

```bash
# Clone the repository
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

# Install dependencies
conda env create -f environment.yaml
conda activate idm

# Download required model checkpoints
# - Follow the instructions in README.md for model checkpoint setup

# Launch the mobile-optimized server
python gradio_demo/mobile_app.py
```

The server will output a URL like: `Running on local URL:  http://0.0.0.0:7860`

## Mobile App Integration

You can integrate with the Gradio server using any of these methods:

### Method 1: Gradio Client in React Native

```javascript
import React, { useState } from 'react';
import { View, Button, Image, TextInput, ActivityIndicator } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import axios from 'axios';

const TryOnScreen = () => {
  const [modelPhoto, setModelPhoto] = useState(null);
  const [garmentPhoto, setGarmentPhoto] = useState(null);
  const [description, setDescription] = useState('');
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Server URL - replace with your deployed server
  const SERVER_URL = 'https://your-server.com/api/tryon';
  
  const selectModelPhoto = () => {
    launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (!response.didCancel) {
        setModelPhoto(response.assets[0]);
      }
    });
  };
  
  const selectGarmentPhoto = () => {
    launchImageLibrary({ mediaType: 'photo' }, (response) => {
      if (!response.didCancel) {
        setGarmentPhoto(response.assets[0]);
      }
    });
  };
  
  const submitTryOn = async () => {
    if (!modelPhoto || !garmentPhoto || !description) {
      alert('Please provide all required inputs');
      return;
    }
    
    setLoading(true);
    
    const formData = new FormData();
    
    // Create the image editor format that Gradio expects
    const modelData = {
      background: {
        url: modelPhoto.uri,
        type: modelPhoto.type,
        name: modelPhoto.fileName,
      },
      layers: null,
    };
    
    formData.append('inputs', JSON.stringify(modelData));
    formData.append('garment', {
      uri: garmentPhoto.uri,
      type: garmentPhoto.type,
      name: garmentPhoto.fileName,
    });
    formData.append('description', description);
    formData.append('auto_mask', true);
    formData.append('auto_crop', true);
    formData.append('denoise_steps', 25);
    formData.append('seed', 42);
    
    try {
      const response = await axios.post(SERVER_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setResultImage(response.data.outputs[0]);
      setLoading(false);
    } catch (error) {
      console.error('Error:', error);
      setLoading(false);
      alert('Failed to process try-on request');
    }
  };
  
  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Button title="Select Your Photo" onPress={selectModelPhoto} />
      {modelPhoto && (
        <Image 
          source={{ uri: modelPhoto.uri }} 
          style={{ width: '100%', height: 300, marginVertical: 10 }} 
        />
      )}
      
      <Button title="Select Garment Photo" onPress={selectGarmentPhoto} />
      {garmentPhoto && (
        <Image 
          source={{ uri: garmentPhoto.uri }} 
          style={{ width: '100%', height: 300, marginVertical: 10 }} 
        />
      )}
      
      <TextInput
        placeholder="Describe the garment (e.g., Blue T-shirt with round neck)"
        value={description}
        onChangeText={setDescription}
        style={{ borderWidth: 1, padding: 10, marginVertical: 10 }}
      />
      
      <Button title="Try It On" onPress={submitTryOn} disabled={loading} />
      
      {loading && <ActivityIndicator size="large" style={{ marginVertical: 20 }} />}
      
      {resultImage && (
        <View>
          <Text>Result:</Text>
          <Image 
            source={{ uri: resultImage }} 
            style={{ width: '100%', height: 300, marginVertical: 10 }} 
          />
        </View>
      )}
    </View>
  );
};

export default TryOnScreen;
```

### Method 2: Flutter Integration

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';

class TryOnScreen extends StatefulWidget {
  @override
  _TryOnScreenState createState() => _TryOnScreenState();
}

class _TryOnScreenState extends State<TryOnScreen> {
  File? modelPhoto;
  File? garmentPhoto;
  String description = '';
  String? resultImageUrl;
  bool isLoading = false;
  
  final ImagePicker _picker = ImagePicker();
  
  // Server URL - replace with your deployed server
  final String SERVER_URL = 'https://your-server.com/api/tryon';
  
  Future<void> _selectModelPhoto() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        modelPhoto = File(image.path);
      });
    }
  }
  
  Future<void> _selectGarmentPhoto() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        garmentPhoto = File(image.path);
      });
    }
  }
  
  Future<void> _submitTryOn() async {
    if (modelPhoto == null || garmentPhoto == null || description.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please provide all required inputs')),
      );
      return;
    }
    
    setState(() {
      isLoading = true;
    });
    
    var request = http.MultipartRequest('POST', Uri.parse(SERVER_URL));
    
    // Add model photo
    request.files.add(await http.MultipartFile.fromPath(
      'model_photo', modelPhoto!.path,
    ));
    
    // Add garment photo
    request.files.add(await http.MultipartFile.fromPath(
      'garment_photo', garmentPhoto!.path,
    ));
    
    // Add other parameters
    request.fields['description'] = description;
    request.fields['auto_mask'] = 'true';
    request.fields['auto_crop'] = 'true';
    request.fields['denoise_steps'] = '25';
    request.fields['seed'] = '42';
    
    try {
      var response = await request.send();
      
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var data = jsonDecode(responseData);
        
        setState(() {
          resultImageUrl = data['output_url'];
          isLoading = false;
        });
      } else {
        setState(() {
          isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to process try-on request')),
        );
      }
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Virtual Try-On'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: _selectModelPhoto,
              child: Text('Select Your Photo'),
            ),
            if (modelPhoto != null)
              Padding(
                padding: EdgeInsets.symmetric(vertical: 8.0),
                child: Image.file(
                  modelPhoto!,
                  height: 300,
                  fit: BoxFit.contain,
                ),
              ),
            
            ElevatedButton(
              onPressed: _selectGarmentPhoto,
              child: Text('Select Garment Photo'),
            ),
            if (garmentPhoto != null)
              Padding(
                padding: EdgeInsets.symmetric(vertical: 8.0),
                child: Image.file(
                  garmentPhoto!,
                  height: 300,
                  fit: BoxFit.contain,
                ),
              ),
            
            TextField(
              decoration: InputDecoration(
                labelText: 'Garment Description',
                hintText: 'e.g., Blue T-shirt with round neck',
              ),
              onChanged: (value) {
                setState(() {
                  description = value;
                });
              },
            ),
            
            SizedBox(height: 16.0),
            
            ElevatedButton(
              onPressed: isLoading ? null : _submitTryOn,
              child: isLoading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text('Try It On'),
            ),
            
            if (resultImageUrl != null)
              Padding(
                padding: EdgeInsets.only(top: 16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Result:', style: TextStyle(fontWeight: FontWeight.bold)),
                    SizedBox(height: 8.0),
                    Image.network(
                      resultImageUrl!,
                      height: 300,
                      fit: BoxFit.contain,
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
```

### Method 3: SwiftUI Integration (iOS)

```swift
import SwiftUI
import UIKit

struct TryOnView: View {
    @State private var modelImage: UIImage?
    @State private var garmentImage: UIImage?
    @State private var description: String = ""
    @State private var resultImage: UIImage?
    @State private var isLoading: Bool = false
    @State private var showModelImagePicker: Bool = false
    @State private var showGarmentImagePicker: Bool = false
    
    let serverURL = "https://your-server.com/api/tryon"
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Button("Select Your Photo") {
                    showModelImagePicker = true
                }
                
                if let modelImage = modelImage {
                    Image(uiImage: modelImage)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                }
                
                Button("Select Garment Photo") {
                    showGarmentImagePicker = true
                }
                
                if let garmentImage = garmentImage {
                    Image(uiImage: garmentImage)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                }
                
                TextField("Describe the garment (e.g., Blue T-shirt with round neck)", text: $description)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)
                
                Button("Try It On") {
                    submitTryOn()
                }
                .disabled(isLoading || modelImage == nil || garmentImage == nil || description.isEmpty)
                
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                }
                
                if let resultImage = resultImage {
                    Text("Result:")
                        .font(.headline)
                    Image(uiImage: resultImage)
                        .resizable()
                        .scaledToFit()
                        .frame(height::300)
                }
            }
            .padding()
        }
        .sheet(isPresented: $showModelImagePicker) {
            ImagePicker(selectedImage: $modelImage)
        }
        .sheet(isPresented: $showGarmentImagePicker) {
            ImagePicker(selectedImage: $garmentImage)
        }
    }
    
    func submitTryOn() {
        guard let modelImage = modelImage, let garmentImage = garmentImage, !description.isEmpty else {
            return
        }
        
        isLoading = true
        
        // Create URL request
        guard let url = URL(string: serverURL) else {
            isLoading = false
            return
        }
        
        // Create multipart request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add model image
        if let modelData = modelImage.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"model_photo\"; filename=\"model.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(modelData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // Add garment image
        if let garmentData = garmentImage.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"garment_photo\"; filename=\"garment.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(garmentData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // Add description
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"description\"\r\n\r\n".data(using: .utf8)!)
        body.append(description.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add other parameters
        let params = [
            "auto_mask": "true",
            "auto_crop": "true",
            "denoise_steps": "25",
            "seed": "42"
        ]
        
        for (key, value) in params {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n".data(using: .utf8)!)
            body.append(value.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    print("Error: \(error)")
                    return
                }
                
                guard let data = data else {
                    return
                }
                
                // Process response - assuming the server returns an image directly
                if let image = UIImage(data: data) {
                    resultImage = image
                }
            }
        }.resume()
    }
}

// Image Picker implementation
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}
```

## API Documentation

The Gradio API endpoint at `/api/tryon` accepts the following parameters:

| Parameter      | Type    | Description                                        |
|----------------|---------|----------------------------------------------------|
| model_photo    | File    | The user's photo                                   |
| garment_photo  | File    | The garment image                                  |
| description    | String  | Text description of the garment                    |
| auto_mask      | Boolean | Whether to generate mask automatically             |
| auto_crop      | Boolean | Whether to auto-crop and resize the image          |
| denoise_steps  | Integer | Number of denoising steps (20-40, higher = better) |
| seed           | Integer | Random seed for reproducibility                    |

## Performance Considerations

- The server requires significant GPU resources for inference
- Mobile devices will only send images and receive results
- Consider implementing a queue system for high-traffic scenarios
- Optimize the image sizes sent from mobile devices to reduce bandwidth usage

## Security Considerations

- Implement proper authentication for your API
- Use HTTPS for all communications
- Consider implementing rate limiting to prevent abuse
- Add server-side validation for all user inputs

## Troubleshooting

If you encounter issues:

1. Check server logs for error messages
2. Verify the server is properly running with GPU acceleration
3. Ensure all required model checkpoints are correctly installed
4. Test the API endpoint directly using tools like Postman before mobile integration
5. Verify network connectivity between the mobile app and server

## Additional Resources

- [Gradio API Documentation](https://gradio.app/docs/)
- [IDM-VTON GitHub Repository](https://github.com/yisol/IDM-VTON)
- [Original IDM-VTON Paper](https://arxiv.org/abs/2403.05139) 