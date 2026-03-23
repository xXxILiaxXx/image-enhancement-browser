import { useEffect, useState } from "react";
import { enhancePhoto } from "../features/enhancement/enhancementService";
import "../styles/app.css";

import {
  createTask,
  subscribeToTask,
  getTaskResult,
  cancelTask,
} from "../features/tasks/taskService";

const translations = {
  ru: {
    title: "ML-модель для улучшения изображений",
    subtitle:
      "Улучшение яркости, контраста и цветности прямо в браузере с помощью обученной модели машинного обучения",
    chooseFile: "Выбрать файл",
    noFile: "Файл не выбран",
    enhance: "Улучшить фото",
    processing: "Обработка изображения...",
    queued: "Задача поставлена в очередь",
    creating: "Создание задачи...",
    idle: "Загрузите фото",
    done: "Готово",
    cancelled: "Задача отменена",
    source: "Исходное",
    result: "Улучшенное",
    download: "Скачать фотографию с улучшенными результатами",
    brightness: "Яркость",
    contrast: "Контраст",
    saturation: "Насыщенность",
    error: "Ошибка обработки",
    aboutTitle: "О модели",
    aboutText:
      "Данная модель разработана Илиевым Илией Ивелиновичем. Модель была обучена на датасете из 5000 RAW-фотографий с различными характеристиками яркости, контраста и цветности. Благодаря этому система способна подбирать параметры коррекции и выдавать улучшенный результат для Ваших изображений.",
    cancel: "Отменить",
    themeLight: "С.Т",
    themeDark: "Т.Т",
  },
  eng: {
    title: "ML model for image enhancement",
    subtitle:
      "Improve brightness, contrast and color directly in the browser using a trained machine learning model.",
    chooseFile: "Choose file",
    noFile: "No file selected",
    enhance: "Enhance image",
    processing: "Processing image...",
    queued: "Task queued",
    creating: "Creating task...",
    idle: "Upload a photo",
    done: "Done",
    cancelled: "Task cancelled",
    source: "Original",
    result: "Enhanced",
    download: "Download",
    brightness: "Brightness",
    contrast: "Contrast",
    saturation: "Saturation",
    error: "Processing error",
    aboutTitle: "About the model",
    aboutText:
      "This model was developed by Iliev Ilia Ivelinovich. It was trained on a dataset of 5000 RAW photographs with different brightness, contrast and color characteristics. As a result, the system is able to estimate correction parameters and produce an enhanced output for user images.",
    cancel: "Cancel",
    themeLight: "L.T",
    themeDark: "D.T",
  },
};

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState("");
  const [correctedUrl, setCorrectedUrl] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [statusKey, setStatusKey] = useState("idle");
  const [statusText, setStatusText] = useState("");

  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState("light");
  const [language, setLanguage] = useState("ru");
  const [taskId, setTaskId] = useState(null);
  const [taskProgress, setTaskProgress] = useState(0);

  const t = translations[language];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (statusKey === "idle") setStatusText(t.idle);
    if (statusKey === "done") setStatusText(t.done);
    if (statusKey === "cancelled") setStatusText(t.cancelled);
  }, [language, statusKey]);

  async function handleEnhance() {
    if (!selectedFile) return;

    try {
      setLoading(true);
      setPrediction(null);
      setOriginalUrl("");
      setCorrectedUrl("");
      setTaskProgress(0);

      setStatusKey("queued");
      setStatusText(t.creating);

      const newTaskId = await createTask(selectedFile, enhancePhoto);
      setTaskId(newTaskId);

      const unsubscribe = subscribeToTask(newTaskId, (task) => {
        setTaskProgress(task.progress);

        if (task.status === "queued") {
          setStatusKey("queued");
          setStatusText(t.queued);
        } else if (task.status === "processing") {
          setStatusKey("processing");
          setStatusText(`${t.processing} ${task.progress}%`);
        } else if (task.status === "done") {
          const result = getTaskResult(newTaskId);

          if (result) {
            setOriginalUrl(result.originalUrl);
            setCorrectedUrl(result.correctedUrl);
            setPrediction(result.prediction);
          }

          setStatusKey("done");
          setStatusText(t.done);
          setLoading(false);
          unsubscribe();
        } else if (task.status === "error") {
          setStatusKey("error");
          setStatusText(`${t.error}: ${task.error}`);
          setLoading(false);
          unsubscribe();
        } else if (task.status === "cancelled") {
          setStatusKey("cancelled");
          setStatusText(t.cancelled);
          setLoading(false);
          unsubscribe();
        }
      });
    } catch (error) {
      console.error(error);
      setStatusKey("error");
      setStatusText(`${t.error}: ${error?.message || error}`);
      setLoading(false);
    }
  }

  function handleThemeToggle() {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }

  function handleCancelTask() {
    if (!taskId) return;
    cancelTask(taskId);
  }

  function handleLanguageToggle() {
    setLanguage((prev) => (prev === "ru" ? "eng" : "ru"));
  }

  return (
  <div className="app">
    <div className="card">
      <div className="topbar">
        <div className="topbar-controls">
          <div className="info-hover-wrap">
            <button className="info-button" type="button" aria-label="About model">
              ?
            </button>

            <div className="info-popover">
              <div className="info-popover-title">{t.aboutTitle}</div>
              <div className="info-popover-text">{t.aboutText}</div>
            </div>
          </div>

          <button
            className="lang-toggle"
            onClick={handleLanguageToggle}
            type="button"
            aria-label="Switch language"
          >
            <span className={`lang-toggle-thumb ${language === "eng" ? "eng" : ""}`}>
              {language === "ru" ? "RU" : "EN"}
            </span>
          </button>

          <button
            className="theme-toggle"
            onClick={handleThemeToggle}
            type="button"
            aria-label="Toggle theme"
          >
            <span className={`theme-toggle-thumb ${theme === "light" ? "light" : ""}`}>
              {theme === "dark" ? t.themeDark : t.themeLight}
            </span>
          </button>
        </div>
      </div>

      <h1>{t.title}</h1>
      <p className="subtitle">{t.subtitle}</p>

      <div className="controls">
        <input
          id="file-input"
          className="file-input"
          type="file"
          accept="image/*,.heic,.heif,.bmp"
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
        />

        <label htmlFor="file-input" className="file-input-label">
          {t.chooseFile}
        </label>

        <span className="file-name">
          {selectedFile
            ? language === "ru"
              ? "Файл выбран"
              : "File selected"
            : t.noFile}
        </span>

        <button onClick={handleEnhance} disabled={!selectedFile || loading}>
          {loading ? t.processing : t.enhance}
        </button>

        {loading && (
          <button type="button" onClick={handleCancelTask}>
            {t.cancel}
          </button>
        )}
      </div>

      <div className={`status ${loading ? "status-processing" : ""}`}>
        {statusText}
      </div>

      {loading && (
        <div className="progress-wrap">
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{ width: `${taskProgress}%` }}
            />
          </div>
          <div className="progress-text">{taskProgress}%</div>
        </div>
      )}

      {!loading && prediction && (
        <div className="prediction prediction-inline">
          <span>
            <strong>{t.brightness}:</strong> {prediction.brightness.toFixed(4)}
          </span>
          <span>
            <strong>{t.contrast}:</strong> {prediction.contrast.toFixed(4)}
          </span>
          <span>
            <strong>{t.saturation}:</strong> {prediction.saturation.toFixed(4)}
          </span>
        </div>
      )}

      {!loading && (originalUrl || correctedUrl) && (
        <>
          <div className="images">
            {originalUrl && (
              <div className="image-card">
                <h3>{t.source}</h3>
                <img src={originalUrl} alt="original" />
              </div>
            )}

            {correctedUrl && (
              <div className="image-card">
                <h3>{t.result}</h3>
                <img src={correctedUrl} alt="corrected" />
              </div>
            )}
          </div>

          {correctedUrl && (
            <div className="download-wrap">
              <a
                className="download-link"
                href={correctedUrl}
                download="enhanced.jpg"
              >
                {t.download}
              </a>
            </div>
          )}
        </>
      )}
    </div>
  </div>
);
}