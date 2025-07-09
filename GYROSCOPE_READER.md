# PhantomStroke - Gyroscope Data Reader

A web-based gyroscope data reader that captures 3-axis angular velocity data at 100 Hz using JavaScript.

## Features

### 🔧 API Support
- **Primary**: Generic Sensor API (Gyroscope) for high-precision readings at 100 Hz
- **Fallback**: DeviceOrientationEvent for lower precision when Generic Sensor API is unavailable
- Automatic API detection and graceful fallback

### 📊 Data Capture
- Real-time X, Y, Z angular velocity readings (rad/s)
- Configurable sampling frequency (1-1000 Hz)
- Configurable buffer size (10-10,000 samples)
- Live data visualization and logging

### 💾 Data Management
- Real-time display of gyroscope values
- Data logging with timestamps
- Save to browser local storage
- Export data as JSON file
- Automatic buffer management

### 📈 Monitoring
- Live sample rate display
- Running time counter
- Data point counter
- API source indicator

## Usage

### Quick Start
1. Open `index.html` in a modern web browser (Chrome, Firefox, Safari, Edge)
2. Allow sensor permissions when prompted
3. Click "▶️ Start Reading" to begin data capture
4. View real-time X, Y, Z values and logs
5. Use "⏹️ Stop Reading" to stop capture

### Settings
- **Sampling Frequency**: Adjust from 1-1000 Hz (default: 100 Hz)
- **Buffer Size**: Set maximum stored samples (default: 1000)

### Data Export
- **Local Storage**: Click "💾 Save to Local Storage" to persist data
- **Export File**: Click "📁 Export Data" to download JSON file with metadata

## Browser Compatibility

### Generic Sensor API (High Precision)
- Chrome 67+
- Edge 79+
- Requires HTTPS in production
- Requires user permission

### DeviceOrientationEvent (Fallback)
- All modern browsers
- iOS Safari (with permission)
- Android Chrome
- Lower precision but broader compatibility

## Security Requirements

- **HTTPS**: Required for Generic Sensor API in production
- **User Permission**: Both APIs require explicit user consent
- **Secure Context**: Must be served over HTTPS or localhost

## Technical Details

### Data Format
```json
{
  "metadata": {
    "timestamp": "2025-01-01T00:00:00.000Z",
    "sampleCount": 1000,
    "frequency": 100,
    "source": "Generic Sensor",
    "userAgent": "..."
  },
  "data": [
    {
      "timestamp": 1640995200000,
      "x": 0.123,
      "y": -0.456,
      "z": 0.789
    }
  ]
}
```

### API Flow
1. Check for `Gyroscope` in `window` object
2. If available, use Generic Sensor API with requested frequency
3. If not available, fallback to `DeviceOrientationEvent`
4. Request appropriate permissions
5. Start data capture and buffering
6. Provide real-time display and logging

## Use Cases

- Motion analysis and gesture recognition
- Vibration detection and analysis
- Device orientation monitoring
- Research and data collection
- Security research (as outlined in project documentation)

## Error Handling

The application handles common scenarios:
- No gyroscope hardware available
- Permission denied by user
- API not supported by browser
- Network connectivity issues
- Data buffer overflow

## Development

### File Structure
```
PhantomStroke/
├── index.html          # Main gyroscope reader application
├── README.md           # This documentation
├── flowchart.md        # Technical flowchart documentation
├── flow.md             # Process flow diagram
└── LICENSE             # License information
```

### Local Development
```bash
# Serve locally (required for sensor APIs)
python3 -m http.server 8080
# or
npx serve .
```

Visit `http://localhost:8080` to test the application.

## License

See [LICENSE](LICENSE) file for details.