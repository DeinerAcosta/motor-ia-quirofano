class StaticVerifier:
    def __init__(self):
        pass 

    def verificar_bandeja(self, detected_classes, checklist_esperado):
        """
        Compara lo detectado por YOLO con el checklist del Excel.
        
        Args:
            detected_classes (list): Lista de nombres de clases detectadas por YOLO (ej. ['tijera', 'pinza', 'pinza']).
            checklist_esperado (list of dict): El checklist que viene de tu app.py (ej. [{'tipo_instrumento': 'Tijera', 'cantidad': 1}, ...])
        
        Returns:
            list of dict: El checklist actualizado con estados 'Detectado'/'Faltante'.
        """

        checklist_resultado = [item.copy() for item in checklist_esperado]
        

        conteo_detectado = {}
        for obj in detected_classes:
            nombre = obj.lower().strip()
            conteo_detectado[nombre] = conteo_detectado.get(nombre, 0) + 1

        for item in checklist_resultado:
            nombre_esperado = str(item['tipo_instrumento']).lower().strip()
            cantidad_esperada = item.get('cantidad_esperada', 1) 
            
            cantidad_encontrada = 0
            for nombre_detectado, cantidad in conteo_detectado.items():
                if nombre_esperado in nombre_detectado or nombre_detectado in nombre_esperado:
                    cantidad_encontrada += cantidad
            
            if cantidad_encontrada >= cantidad_esperada:
                 item['estado'] = 'Completo'
                 item['cantidad_encontrada'] = cantidad_encontrada
            elif cantidad_encontrada > 0:
                 item['estado'] = f'Incompleto ({cantidad_encontrada}/{cantidad_esperada})'
                 item['cantidad_encontrada'] = cantidad_encontrada
            else:
                 item['estado'] = 'Faltante'
                 item['cantidad_encontrada'] = 0
                 
        return checklist_resultado