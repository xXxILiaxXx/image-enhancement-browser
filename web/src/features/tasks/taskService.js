const tasks = new Map();

function generateTaskId() {
  return `task_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function emitTaskUpdate(task) {
  task.listeners.forEach((listener) => listener({
    id: task.id,
    status: task.status,
    progress: task.progress,
    error: task.error,
  }));
}

function updateTask(task, patch) {
  Object.assign(task, patch);
  emitTaskUpdate(task);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function subscribeToTask(taskId, listener) {
  const task = tasks.get(taskId);
  if (!task) return () => {};

  task.listeners.add(listener);

  listener({
    id: task.id,
    status: task.status,
    progress: task.progress,
    error: task.error,
  });

  return () => {
    const currentTask = tasks.get(taskId);
    if (currentTask) {
      currentTask.listeners.delete(listener);
    }
  };
}

export function getTaskStatus(taskId) {
  const task = tasks.get(taskId);
  if (!task) return null;

  return {
    id: task.id,
    status: task.status,
    progress: task.progress,
    error: task.error,
  };
}

export function getTaskResult(taskId) {
  const task = tasks.get(taskId);
  if (!task) return null;
  return task.result ?? null;
}

export function cancelTask(taskId) {
  const task = tasks.get(taskId);
  if (!task) return { success: false };

  if (task.status === "done" || task.status === "error" || task.status === "cancelled") {
    return { success: false };
  }

  task.cancelRequested = true;
  updateTask(task, {
    status: "cancelled",
    progress: 0,
    error: null,
  });

  return { success: true };
}

export async function createTask(file, enhancePhotoFn) {
  const id = generateTaskId();

  const task = {
    id,
    file,
    status: "queued",
    progress: 0,
    error: null,
    result: null,
    cancelRequested: false,
    listeners: new Set(),
  };

  tasks.set(id, task);
  emitTaskUpdate(task);

  (async () => {
    try {
      updateTask(task, { status: "queued", progress: 5 });
      await wait(120);

      if (task.cancelRequested) return;

      updateTask(task, { status: "processing", progress: 15 });
      await wait(150);

      if (task.cancelRequested) return;

      updateTask(task, { status: "processing", progress: 35 });
      await wait(160);

      if (task.cancelRequested) return;

      updateTask(task, { status: "processing", progress: 55 });
      await wait(160);

      if (task.cancelRequested) return;

      const result = await enhancePhotoFn(file);

      if (task.cancelRequested) return;

      updateTask(task, { status: "processing", progress: 85 });
      await wait(120);

      if (task.cancelRequested) return;

      task.result = result;

      updateTask(task, {
        status: "done",
        progress: 100,
        error: null,
      });
    } catch (error) {
      updateTask(task, {
        status: "error",
        progress: 0,
        error: error?.message || String(error),
      });
    }
  })();

  return id;
}