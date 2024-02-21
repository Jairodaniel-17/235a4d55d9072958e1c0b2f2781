# para ejecutar usar el comando: python -m jobs.job_1_espera_equipos


# funcional en ppython
def buscar_clave(diccionario, lista_oraciones):
    for oracion in lista_oraciones:
        oracion = oracion.lower()
        for clave, valores in diccionario.items():
            for valor in valores:
                if valor.lower() in oracion:
                    return clave
    return None


# Diccionario de palabras clave
diccionario = {
    "Router": {"ONT (FTTH)", "ONT"},
    "Switch": {"Switch"},
    "DECO IPTV": {"DECO IPTV"},
    "DECO HD": {"DECO HD"},
    "DVR": {"DVR"},
    "EMTA": {"EMTA 3.1", "EMTA"},
    "MTA": {"ARRIS TG2482"},
    "Repetidor": {"INT-REPETIDOR WF-808 CIG"},
    "Kit de fibra": {"Conectores ópticos y ONT", "MATERIAL FTTH.", "ftth"},
    "Conector óptico": {
        "Cónectores FTTH",
        "Conector logico",
        "Conectores logicos",
        "Conector óptico",
    },
    "DROP 50-80": {
        "DROP 50-80",
        "DROP 50 -80",
        "DROP 50- 80",
        "DROP 50 - 80",
        "DROP 50-80 .",
        "DROP 50-80.",
    },
    "DROP 220": {"DROP 220", "drop 220", "DROP 220.", "drop 220."},
    "Arris": {"Arris 3.1", "Arris TG3442A"},
    "DROP 150": {"DROP 150", "drop 150", "DROP 150.", "drop 150."},
    "DROP 100": {"DROP 100", "drop 100", "DROP 100.", "drop 100."},
    "DROP 80": {"DROP 80", "drop 80", "DROP 80.", "drop 80."},
    "DROP 50": {"DROP 50", "drop 50", "DROP 50.", "drop 50."},
}

lista_oraciones = [
    "Agendado en ETADirect EMTA 3.1",
    "EN ESPERA DE QUIPOS MESA DE PROGRAMACION sin Stock de equipos Modelo: Stock equipos ftth Realizado por: Leidy Ramirez",
]

resultado = buscar_clave(diccionario, lista_oraciones)
print("Claves encontradas:", resultado)
