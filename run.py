import argparse
from pathlib import Path
from api import create_app
from core.engine import engine

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description="traffic-plates ANPR web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip loading models at startup (loads on first request)")
    args = p.parse_args()

    app = create_app()

    print("=" * 60)
    print(f"traffic-plates ANPR web app")
    print(f"  YOLO11 model : {HERE / 'yolo11_plate.pt'}")
    print(f"  Awiros OCR   : {HERE / 'awiros_anpr' / 'model.safetensors'}")
    print(f"  Listening on : http://{args.host}:{args.port}")
    print("=" * 60)

    if not args.no_warmup:
        engine.warmup()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
