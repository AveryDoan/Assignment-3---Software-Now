# Setup and Running

## Virtual Environment Setup

Create and activate a virtual environment:

```bash
cd Assignment-3---Software-Now
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

With venv activated:

```bash
python3 main.py
```

## Docker Setup (Recommended for macOS)

If you encounter OpenCV errors on macOS, use Docker instead:

```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop

# Build the image
docker build -t spot-difference-game .

# Run the container
docker run -it --rm spot-difference-game
```

Or with docker-compose:

```bash
docker-compose up --build
```

## Troubleshooting

### macOS OpenCV Compatibility Issue

If you encounter an error like:
```
macOS 26 (2603) or later required, have instead 16 (1603) !
```

This is a known OpenCV binary compatibility issue on some macOS versions. The code is correct; the issue is with the OpenCV dylib.

**Solutions:**
1. **Use Docker (Recommended)** - Container-based approach avoids system library conflicts
2. **Use Rosetta 2 (Intel emulation)** - Try running Python under Rosetta instead of native ARM64
3. **Install from source** - Compile OpenCV locally for your system
4. **Use a different machine** - The code works correctly on systems with compatible OpenCV binaries (Windows, Linux)

The `requirements.txt` specifies `opencv-python-headless==4.7.0.72`, which is the most compatible version for this setup.

## Code Structure

Each file has an owner and clear responsibilities:

- **main.py** (Avery) - Application entry point
- **game_controller.py** (Avery) - Application orchestrator
- **image_processor.py** (Max) - OpenCV image processing
- **game_logic.py** (Alvi) - Game rules and state
- **gui.py** (Jane) - Tkinter interface
