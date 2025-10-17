# Google Maps API Setup Instructions

## Current Status
✅ **Backend**: Working with fallback system
✅ **Frontend**: Using OpenStreetMap for development mode
⚠️ **Google Maps**: API key needs proper permissions

## To Enable Full Google Street View:

### 1. Go to Google Cloud Console
- Visit: https://console.cloud.google.com/
- Create a new project or select existing project

### 2. Enable Required APIs
Navigate to "APIs & Services" → "Library" and enable:
- ✅ **Street View Static API**
- ✅ **Maps Embed API** 
- ✅ **Maps JavaScript API**

### 3. Create/Configure API Key
- Go to "APIs & Services" → "Credentials"
- Click "Create Credentials" → "API Key"
- Copy the generated API key

### 4. Configure API Key Restrictions (Recommended)
- Click on your API key to edit
- Under "API restrictions", select "Restrict key"
- Choose the APIs you enabled above
- Under "Application restrictions", add your domains:
  - `localhost:3000` (for development)
  - Your production domain

### 5. Update Environment Files
Replace the API key in these files:
- `/backend/.env`: `GOOGLE_MAPS_API_KEY=YOUR_NEW_KEY`
- `/frontend/.env.local`: `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=YOUR_NEW_KEY`

### 6. Restart Servers
```bash
# Restart backend
pkill -f "uvicorn.*8001"
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# Restart frontend  
cd frontend && npm run dev
```

## Current Development Mode Features:
- 🗺️ **OpenStreetMap**: Shows location context
- 📍 **Coordinates**: Displays exact lat/lng
- 🎯 **Fair Game**: Both AI and human see same location
- 🔄 **Fallbacks**: System works without Google Maps API

## After API Setup:
- 🌍 **Interactive Street View**: Full 360° navigation
- 🚗 **Tesla-style AI**: Colored analysis overlays
- 🎮 **Complete Experience**: Just like GeoGuessr

The system is fully functional in development mode. The Google Maps API will enhance the experience with interactive Street View!
