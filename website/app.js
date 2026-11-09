// Progressive enhancement: the page and every document link work without JS.
const modes = {
  observe: {
    title: 'Let the world correct the model.',
    text: 'Real depth, touch, and joint history update the recurrent belief. This observation path is part of the verified original-policy replay.'
  },
  imagine: {
    title: 'Let a short future inform the action.',
    text: 'A separate SM model predicts physical trajectories for BPTT. WSM policy context must evolve through a validated predicted-observation or feature bridge. SM has learned sustained Sharpa single-object rotation from scratch. Matched PPO learning speed and the benefit of prior context remain under investigation.'
  }
};
const modeButtons = document.querySelectorAll('[data-mode]');
modeButtons.forEach(button => {
  button.addEventListener('click', () => {
    const mode = modes[button.dataset.mode];
    if (!mode) return;
    modeButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    const panel = document.querySelector('#mode-copy');
    panel.querySelector('h3').textContent = mode.title;
    panel.querySelector('p').textContent = mode.text;
  });
});
