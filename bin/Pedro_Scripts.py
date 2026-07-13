#!/usr/bin/env python

def get_volcano_info(volcano_name, volcanoes_file):
    try:
        with open(volcanoes_file, "r", encoding="utf-8") as vf:
            next(vf)  # saltar header
            for line in vf:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"'):
                    parts = line.split()
                    name_tokens = []
                    for token in parts:
                        name_tokens.append(token)
                        if token.endswith('"'):
                            break
                    nombre_volcan = " ".join(name_tokens).strip('"')
                    rest = parts[len(name_tokens):]
                    if len(rest) != 3:
                        continue
                    lon = float(rest[0].rstrip(','))
                    lat = float(rest[1].rstrip(','))
                    distancia = float(rest[2].rstrip(','))
                else:
                    parts = line.split()
                    nombre_volcan = parts[0]
                    lon = float(parts[1].rstrip(','))
                    lat = float(parts[2].rstrip(','))
                    distancia = float(parts[3].rstrip(','))
                if nombre_volcan.lower() == volcano_name.strip('"').lower():
                    return nombre_volcan, lon, lat, distancia
    except Exception as e:
        print(f"Error leyendo archivo de volcanes: {e}")
    return None