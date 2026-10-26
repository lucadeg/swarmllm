// The eval tasks in a fixed order (docs/design/harness-light.md C.2).
import tetris from "./tetris.js";
import snake from "./snake.js";
import todo from "./todo.js";
import calculator from "./calculator.js";
import stopwatch from "./stopwatch.js";
import landing from "./landing.js";
import fixBug from "./fix-bug.js";
import fixCrash from "./fix-crash.js";
import addFeature from "./add-feature.js";
import css from "./css.js";
import refactor from "./refactor.js";
import logic from "./logic.js";

export const TASKS = [tetris, snake, todo, calculator, stopwatch, landing, fixBug, fixCrash, addFeature, css, refactor, logic];
export const byId = (id) => TASKS.find((t) => t.id === id);
