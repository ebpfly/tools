# Reflectance Spectrum Explorer

A simple, interactive web UI for exploring and analyzing reflectance spectrum data.

## Features

- **Interactive Visualization**: Plot and explore reflectance spectra with an intuitive canvas-based interface
- **Zoom & Pan**: Use mouse wheel to zoom and click-drag to pan across the spectrum
- **Data Import**: Load your own spectrum data from CSV files
- **Sample Data**: Built-in sample spectrum data for quick testing
- **Real-time Statistics**: View min, max, mean reflectance values and wavelength range
- **Hover Information**: See wavelength and reflectance values as you move your mouse
- **Data Export**: Export spectrum data to CSV format
- **Customizable Display**: Toggle grid and data point markers on/off

## Usage

### Getting Started

1. Open `index.html` in a web browser
2. The application loads with sample spectrum data automatically

### Loading Your Own Data

1. Click "Load Spectrum Data" and select a CSV file
2. CSV format should be: `wavelength,reflectance`
3. Example:
   ```
   380,45.2
   385,46.8
   390,48.5
   ```

### Interactions

- **Zoom**: Scroll mouse wheel while hovering over the plot
- **Pan**: Click and drag to move around the spectrum
- **Reset Zoom**: Click the "Reset Zoom" button to fit all data
- **Hover**: Move mouse over the plot to see wavelength/reflectance values
- **Export**: Click "Export Data" to download current spectrum as CSV

### Display Options

- **Show Grid**: Toggle grid lines on/off
- **Show Markers**: Toggle data point markers on/off

## File Structure

```
.
├── index.html    # Main HTML structure
├── style.css     # Styling and layout
├── script.js     # Interactive functionality
└── README.md     # This file
```

## Technical Details

- Pure HTML/CSS/JavaScript - no external dependencies
- Canvas-based rendering for smooth performance
- Responsive design for different screen sizes
- Sample data covers visible spectrum range (380-780 nm)

## Browser Compatibility

Works in all modern browsers that support:
- HTML5 Canvas
- ES6 JavaScript
- CSS Grid

## License

Open source - feel free to use and modify as needed.
