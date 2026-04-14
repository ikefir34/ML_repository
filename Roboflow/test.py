from inference_sdk import InferenceHTTPClient

# Инициализация клиента для облачного API
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ВАШ_API_KEY"
)

# Выполнение инференса
results = client.infer("image.jpg", model_id="project-name/version")
print(results)



from inference import get_model

# Загрузка модели локально (скачивается один раз)
model = get_model(model_id="project-name/version", api_key="ВАШ_API_KEY")

# Предсказание
results = model.infer(image="image.jpg")
print(results[0].predictions)


from roboflow import Roboflow

rf = Roboflow(api_key="ВАШ_API_KEY")
project = rf.workspace("название_workspace").project("название_проекта")

# Загрузка одного изображения
project.upload("image.jpg", num_retry_attempts=3)