# scripts/download_from_yadisk.py
import requests
import os
from pathlib import Path
import re
import time

def get_direct_download_url(public_url):
    """Получаем прямую ссылку на скачивание из публичной ссылки Яндекс.Диска."""
    
    url = public_url.strip()
    
    # Извлекаем ID файла
    match = re.search(r'(?:disk\.yandex\.ru|yadi\.sk)/(?:d|i)/([a-zA-Z0-9_-]+)', url)
    if not match:
        print(f"❌ Не удалось извлечь ID из ссылки: {url}")
        return None
    
    file_id = match.group(1)
    print(f"🔍 Извлечён ID: {file_id}")
    
    # Заголовки браузера
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
    }
    
    # Пробуем получить ссылку через официальный API
    api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={file_id}"
    
    try:
        print(f"🌐 Запрос к API: {api_url[:80]}...")
        response = requests.get(api_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            if 'href' in data:
                print(f"✅ Получена прямая ссылка")
                return data['href']
            else:
                print(f"⚠️ API вернул ответ без 'href': {data}")
        else:
            print(f"⚠️ API вернул статус {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Ошибка при запросе к API: {e}")
    
    # Фоллбэк: пробуем прямую ссылку с ?dl=1
    fallback_url = f"https://yadi.sk/d/{file_id}?dl=1"
    print(f"🔄 Пробуем фоллбэк: {fallback_url}")
    return fallback_url


def download_file(url, save_path):
    """Скачивает файл по ссылке."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    try:
        # Создаём папку
        save_dir = os.path.dirname(save_path)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        filename = os.path.basename(save_path)
        print(f"⬇️ Скачиваем {filename}...")
        
        # Скачиваем
        response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        
        if response.status_code != 200:
            print(f"❌ Ошибка HTTP {response.status_code}")
            return False
        
        # Проверяем Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type and filename.endswith('.csv'):
            print(f"❌ Получен HTML вместо файла (возможно, капча)")
            return False
        
        # Сохраняем
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        size = os.path.getsize(save_path)
        
        # Дополнительная проверка для CSV
        if filename.endswith('.csv') and size < 2000:
            with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(200).lower()
                if '<!doctype html>' in content or '<html' in content or 'captcha' in content:
                    print(f"❌ Файл содержит HTML/капчу, удаляем...")
                    os.remove(save_path)
                    return False
        
        print(f"✅ Сохранено: {save_path} ({size / 1024:.1f} КБ)")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


# ==================== НАСТРОЙКИ ====================
Path("data/raw").mkdir(parents=True, exist_ok=True)

files = {
    "data/raw/bank.csv": "https://disk.yandex.ru/i/mLoocgbCvlXXbg",
    "data/raw/adult.csv": "https://disk.yandex.ru/d/MrUQwowQjElARQ",
}

# ==================== ЗАПУСК ====================
print("🚀 Начинаем скачивание...\n")

success = 0
for save_path, public_url in files.items():
    print(f"\n📥 Обработка: {save_path}")
    
    direct_url = get_direct_download_url(public_url)
    if not direct_url:
        continue
    
    if download_file(direct_url, save_path):
        success += 1
    
    time.sleep(3)  # Пауза, чтобы не блокировали

print(f"\n🎉 Готово! Успешно скачано: {success}/{len(files)}")

# Проверка результата
if Path("data/raw").exists():
    print(f"\n📁 Содержимое data/raw/:")
    for f in Path("data/raw").iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"   • {f.name} ({size_kb:.1f} КБ)")