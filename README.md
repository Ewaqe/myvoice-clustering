# myvoice-clustering
Чтобы запустить приложение введите:
```
pip install -r requirements.txt
```
```
flask run
```

OS: linux, windows

Приложение имеет API Endpoint ```/uploadFile```, который на вход принимает json файл формата файлов из директории all
На выходе приложение отправляет JSON с кластерами.

Для кластеризации текстовые данные проходят предварительную обработку: лемматизируются при помощи библиотеки [rulemma](https://github.com/Koziev/rulemma), исправляются простые грамматические и орфографические ошибки при помощи библиотеки [autocorrect](https://github.com/filyp/autocorrect), фильтруются сообщения с обсценной лексикой. Затем полученные данные векторизируются при помощи библиотеки [Navec](https://github.com/natasha/navec) и объединяются в кластеры при помощи алгоритма AgglomerativeClusterization
