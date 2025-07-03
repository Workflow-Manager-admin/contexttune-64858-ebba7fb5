from fastapi import FastAPI, HTTPException, Depends, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import sqlite3

# --- App Metadata and FastAPI Setup ---
app = FastAPI(
    title="MuseMap Backend API",
    description="Personalized music/playlist discovery and recommendation based on context—activity, location, language, and time.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User authentication"},
        {"name": "profile", "description": "Profile and preferences"},
        {"name": "music", "description": "Music and playlist APIs"},
        {"name": "map", "description": "Map-based exploration APIs"},
        {"name": "misc", "description": "Miscellaneous and health endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Database Config: SQLite ---
DB_PATH = "musemap.db"

def get_db_conn():
    """Obtain a new SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create schema if needed."""
    conn = get_db_conn()
    c = conn.cursor()
    # Users, their hashed passwords, tokens, profiles, and preferences.
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        language TEXT DEFAULT 'en',
        activity TEXT,
        location TEXT,
        time_pref TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    # Saved preferences for filters etc.
    c.execute("""
    CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        key TEXT,
        value TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    # Playlists and tracks associations.
    c.execute("""
    CREATE TABLE IF NOT EXISTS playlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT,
        context TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        artist TEXT,
        album TEXT,
        language TEXT,
        activity TEXT,
        location TEXT,
        url TEXT
    )""")
    # Association table playlist <-> tracks (many-to-many)
    c.execute("""
    CREATE TABLE IF NOT EXISTS playlist_tracks (
        playlist_id INTEGER,
        track_id INTEGER,
        FOREIGN KEY (playlist_id) REFERENCES playlists(id),
        FOREIGN KEY (track_id) REFERENCES tracks(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# --- JWT Auth Setup ---
SECRET_KEY = "CHANGE_THIS_SECRET_KEY_MUSEMAP"  # In production use env var!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 3  # 3 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# --- Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserProfile(UserBase):
    language: str = Field(..., description="Preferred music language")
    activity: Optional[str] = Field(None, description="Recent or preferred activity (e.g., workout)")
    location: Optional[str] = Field(None, description="Default or saved location (city, country)")
    time_pref: Optional[str] = Field(None, description="User's preferred time/context")

class UserInDB(UserProfile):
    id: int
    hashed_password: str

class Preference(BaseModel):
    key: str
    value: str

class PlaylistCreate(BaseModel):
    name: str
    context: Optional[str]
    track_ids: Optional[List[int]] = None

class PlaylistOut(BaseModel):
    id: int
    name: str
    context: Optional[str]
    created_at: str
    tracks: List[Dict[str, Any]]

class TrackCreate(BaseModel):
    title: str
    artist: str
    album: Optional[str]
    language: str
    activity: Optional[str]
    location: Optional[str]
    url: str

class TrackOut(BaseModel):
    id: int
    title: str
    artist: str
    album: Optional[str]
    language: str
    activity: Optional[str]
    location: Optional[str]
    url: str

# --- Utility Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(conn, username):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return row

def get_user_by_email(conn, email):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    return row

def get_user_by_id(conn, user_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    return row

def authenticate_user(conn, username: str, password: str):
    user = get_user_by_username(conn, username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    conn = get_db_conn()
    user = get_user_by_username(conn, username)
    conn.close()
    if user is None:
        raise credentials_exception
    return user

# --- Routes ---

# PUBLIC_INTERFACE
@app.get("/", tags=["misc"], summary="Root Health Check")
def health_check():
    """Return healthy response on root."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.get("/health/db", tags=["misc"], summary="Database Connection Health Check")
def db_health_check():
    """
    Checks if the application can connect to the SQLite database.
    Returns OK if connection and simple query succeed.
    """
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "database": "ok"}
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)

# --- Auth Endpoints ---

# PUBLIC_INTERFACE
@app.post("/auth/register", tags=["auth"], response_model=Token, summary="Register a new user")
def register(user: UserCreate):
    """
    Create a new account. Returns access token.
    """
    conn = get_db_conn()
    if get_user_by_username(conn, user.username) or get_user_by_email(conn, user.email):
        conn.close()
        raise HTTPException(status_code=409, detail="Username or email already registered")
    hashed_password = get_password_hash(user.password)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, email, hashed_password) VALUES (?, ?, ?)",
        (user.username, user.email, hashed_password)
    )
    conn.commit()
    conn.close()
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# PUBLIC_INTERFACE
@app.post("/auth/token", tags=["auth"], response_model=Token, summary="Obtain JWT access token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_conn()
    user = authenticate_user(conn, form_data.username, form_data.password)
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# PUBLIC_INTERFACE
@app.get("/auth/me", tags=["auth"], response_model=UserProfile, summary="Get current user's profile")
async def read_users_me(current_user=Depends(get_current_user)):
    """Returns user profile information."""
    u = current_user
    return UserProfile(
        username=u["username"], email=u["email"], language=u["language"],
        activity=u["activity"], location=u["location"], time_pref=u["time_pref"]
    )

# --- User Profile and Preferences ---

# PUBLIC_INTERFACE
@app.get("/profile/preferences", tags=["profile"], response_model=List[Preference], summary="Get all preferences for user")
async def get_preferences(current_user=Depends(get_current_user)):
    """Returns all saved preferences for user."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM preferences WHERE user_id = ?", (current_user["id"],))
    prefs = [{"key": row["key"], "value": row["value"]} for row in cur.fetchall()]
    conn.close()
    return prefs

# PUBLIC_INTERFACE
@app.post("/profile/preferences", tags=["profile"], response_model=Preference, summary="Set user preference")
async def set_preference(preference: Preference, current_user=Depends(get_current_user)):
    """Set or update a single user preference."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO preferences (user_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value = ?",
        (current_user["id"], preference.key, preference.value, preference.value)
    )
    conn.commit()
    conn.close()
    return preference

# PUBLIC_INTERFACE
@app.patch("/profile", tags=["profile"], response_model=UserProfile, summary="Update profile")
async def update_profile(profile: UserProfile, current_user=Depends(get_current_user)):
    """Update user profile attributes: language, activity, location, time_pref."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET language = ?, activity = ?, location = ?, time_pref = ? WHERE id = ?",
        (profile.language, profile.activity, profile.location, profile.time_pref, current_user["id"])
    )
    conn.commit()
    cur.execute("SELECT * FROM users WHERE id = ?", (current_user["id"],))
    row = cur.fetchone()
    conn.close()
    return UserProfile(
        username=row["username"], email=row["email"], language=row["language"],
        activity=row["activity"], location=row["location"], time_pref=row["time_pref"]
    )

# --- Playlist APIs ---

# PUBLIC_INTERFACE
@app.post("/playlists", tags=["music"], response_model=PlaylistOut, summary="Create a playlist")
async def create_playlist(playlist: PlaylistCreate, current_user=Depends(get_current_user)):
    """
    Create a new playlist associated with the context and selected tracks.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO playlists (user_id, name, context) VALUES (?, ?, ?)",
        (current_user["id"], playlist.name, playlist.context)
    )
    playlist_id = cur.lastrowid
    # If user specified track_ids, link them
    if playlist.track_ids:
        for tid in playlist.track_ids:
            cur.execute("INSERT INTO playlist_tracks (playlist_id, track_id) VALUES (?, ?)", (playlist_id, tid))
    conn.commit()
    # Retrieve playlist info and translate tracks
    tracks_res = []
    if playlist.track_ids:
        qmarks = ",".join(["?"] * len(playlist.track_ids))
        cur.execute(f"SELECT * FROM tracks WHERE id IN ({qmarks})", playlist.track_ids)
        for row in cur.fetchall():
            tracks_res.append(dict(row))
    else:
        tracks_res = []
    cur.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,))
    p = cur.fetchone()
    conn.close()
    return PlaylistOut(
        id=p["id"], name=p["name"], context=p["context"],
        created_at=p["created_at"], tracks=tracks_res,
    )

# PUBLIC_INTERFACE
@app.get("/playlists", tags=["music"], response_model=List[PlaylistOut], summary="List user playlists")
async def list_playlists(current_user=Depends(get_current_user)):
    """
    Returns playlists for the current user.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM playlists WHERE user_id = ? ORDER BY created_at DESC", (current_user["id"],))
    playlists = []
    for row in cur.fetchall():
        cur2 = conn.cursor()
        cur2.execute("SELECT t.* FROM tracks t JOIN playlist_tracks pt ON t.id = pt.track_id WHERE pt.playlist_id = ?", (row["id"],))
        tracks = [dict(t) for t in cur2.fetchall()]
        playlists.append({
            "id": row["id"], "name": row["name"], "context": row["context"],
            "created_at": row["created_at"], "tracks": tracks
        })
    conn.close()
    return playlists

# PUBLIC_INTERFACE
@app.get("/playlists/{playlist_id}", tags=["music"], response_model=PlaylistOut, summary="Get playlist by ID")
async def get_playlist(playlist_id: int, current_user=Depends(get_current_user)):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM playlists WHERE id = ? AND user_id = ?", (playlist_id, current_user["id"]))
    pl = cur.fetchone()
    if not pl:
        conn.close()
        raise HTTPException(404, "Playlist not found")
    cur.execute("SELECT t.* FROM tracks t JOIN playlist_tracks pt ON t.id = pt.track_id WHERE pt.playlist_id = ?", (playlist_id,))
    tracks = [dict(t) for t in cur.fetchall()]
    conn.close()
    return PlaylistOut(id=pl["id"], name=pl["name"], context=pl["context"], created_at=pl["created_at"], tracks=tracks)

# --- Tracks/Music APIs ---

# PUBLIC_INTERFACE
@app.post("/tracks", tags=["music"], response_model=TrackOut, summary="Add a new track to the library")
async def add_track(track: TrackCreate, current_user=Depends(get_current_user)):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tracks (title, artist, album, language, activity, location, url) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (track.title, track.artist, track.album, track.language, track.activity, track.location, track.url)
    )
    track_id = cur.lastrowid
    conn.commit()
    cur.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
    t = cur.fetchone()
    conn.close()
    return TrackOut(**dict(t))

# PUBLIC_INTERFACE
@app.get("/tracks", tags=["music"], response_model=List[TrackOut], summary="List or filter tracks")
async def list_tracks(
    language: Optional[str] = Query(None, description="Filter by language"),
    activity: Optional[str] = Query(None, description="Filter by activity"),
    location: Optional[str] = Query(None, description="Filter by location"),
    search: Optional[str] = Query(None, description="Search by title or artist")
):
    """Returns tracks filtered by these fields."""
    conn = get_db_conn()
    cur = conn.cursor()
    conditions, params = [], []
    if language:
        conditions.append("language = ?")
        params.append(language)
    if activity:
        conditions.append("activity = ?")
        params.append(activity)
    if location:
        conditions.append("location = ?")
        params.append(location)
    if search:
        conditions.append("(title LIKE ? OR artist LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    cur.execute(f"SELECT * FROM tracks {where_clause} ORDER BY id DESC", params)
    tracks = [dict(r) for r in cur.fetchall()]
    conn.close()
    return tracks

# --- Context-aware and Map-based playlist/music recommendations ---

# PUBLIC_INTERFACE
@app.get("/recommend", tags=["music"], response_model=List[TrackOut], summary="Get recommended tracks")
async def recommend_tracks(
    activity: Optional[str] = Query(None, description="User activity context"),
    location: Optional[str] = Query(None, description="Location context (city/country)"),
    language: Optional[str] = Query(None, description="Preferred language"),
    time: Optional[str] = Query(None, description="Time context (morning, night, etc)"),
    current_user=Depends(get_current_user)
):
    """
    Returns recommended tracks based on user context (dummy logic: best-matching tracks).
    """
    conn = get_db_conn()
    cur = conn.cursor()
    params, conditions = [], []
    if language:
        conditions.append("language = ?")
        params.append(language)
    if activity:
        conditions.append("activity = ?")
        params.append(activity)
    if location:
        conditions.append("location = ?")
        params.append(location)
    # Time is ignored in filtering for demo, but can be mapped to playlists/tracks later.
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    cur.execute(f"SELECT * FROM tracks {where} ORDER BY id DESC LIMIT 20", params)
    tracks = [dict(t) for t in cur.fetchall()]
    conn.close()
    # In future: could use a real recommendation engine here.
    return tracks

# PUBLIC_INTERFACE
@app.get("/explore/map", tags=["map"], response_model=List[TrackOut], summary="Map-based music exploration")
async def map_music(
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude"),
    radius_km: Optional[int] = Query(50, description="Radius in km"),
    language: Optional[str] = Query(None, description="Music language"),
    activity: Optional[str] = Query(None, description="Filter activity"),
):
    """
    Returns tracks for a given location (dummy logic: 'location' field matches to city/country).
    In a real app, we would geocode coordinates here.
    """
    conn = get_db_conn()
    cur = conn.cursor()
    params, conditions = [], []
    if language:
        conditions.append("language = ?")
        params.append(language)
    if activity:
        conditions.append("activity = ?")
        params.append(activity)
    # For demonstration, location field may store city/country. Future: maps API/geocoding.
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    cur.execute(f"SELECT * FROM tracks {where} ORDER BY id DESC LIMIT 20", params)
    tracks = [dict(t) for t in cur.fetchall()]
    conn.close()
    return tracks

# --- Error Handler ---

@app.exception_handler(sqlite3.Error)
def db_exception_handler(request: Request, exc: sqlite3.Error):
    return JSONResponse(
        status_code=500,
        content={"error": "A database error occurred", "details": str(exc)},
    )

# Notes:
# - In production, avoid hard-coded secrets and use password validation rules.
# - This implementation can be expanded: add track popularity, user play history, and integrate real music/location APIs.
# - Cross-origin is set to allow_all for demo—restrict for production.

# --- End of Backend Implementation ---
