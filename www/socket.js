const socket = new WebSocket("ws://localhost:8000/");
const canvas = document.getElementById("lineplot");
const ctx = canvas.getContext("2d");

const voiceId = 131;

const totalRounds = 5;

// start or playing
let state = "start";
let rounds = 0;
let targetNote = null;

const startButton = document.getElementById("start");

let deltaTimes = [];

let lastStartTime = 0;

function startGame(socket) {
  if (state != "start" && state != "resting") {
    return;
  }

  // Generate target note
  targetNote = Math.floor(Math.random() * 24);

  console.log("Started with target note", targetNote);

  // Send note to server
  socket.send(
    JSON.stringify({
      target_note: targetNote,
    }),
  );

  lastStartTime = performance.now() / 1000.0;

  state = "playing";
}

socket.onopen = function (event) {
  console.log("WebSocket connected!");

  startButton.onclick = () => {
    let utterance = new SpeechSynthesisUtterance("Three, two, one, start!");
    utterance.rate = 0.8;
    utterance.pitch = 1.2;

    utterance.voice = window.speechSynthesis.getVoices()[voiceId];

    window.speechSynthesis.speak(utterance);

    utterance.onend = () => {
      setTimeout(() => startGame(socket), 500);
    };
  };
};

socket.onmessage = function (event) {
  if (state == "playing") {
    // we will receive pulses
    let pulseNote = JSON.parse(event.data)["pulse"];

    if (pulseNote == -1) {
      let finishTime = performance.now() / 1000.0;
      let deltaTime = finishTime - lastStartTime;
      deltaTimes.push(deltaTime);

      if (rounds < totalRounds) {
        state = "resting";
        rounds += 1;
        // Play again :)
        setTimeout(() => {
          startGame(socket);
        }, 2000);
      } else {
        state = "start";
        rounds = 0;

        let averageTime =
          Math.round(
            (deltaTimes.reduce((x, y) => x + y, 0) / totalRounds) * 10,
          ) / 10;

        deltaTimes = [];

        setTimeout(() => {
          let utterance = new SpeechSynthesisUtterance(
            `Finished! Average time: ${averageTime} seconds`,
          );
          utterance.rate = 0.8;
          utterance.pitch = 1.5;

          utterance.voice = window.speechSynthesis.getVoices()[voiceId];

          window.speechSynthesis.speak(utterance);
        }, 500);
      }
    }
  }
};

socket.onclose = function (event) {
  console.log("WebSocket closed.");
};

socket.onerror = function (error) {
  console.error("WebSocket error:", error);
};

function drawMovingLine() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "red";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(currentX, 0);
  ctx.lineTo(currentX, canvas.height);
  ctx.stroke();
}
