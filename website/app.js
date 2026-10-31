// Progressive enhancement: the page and every document link work without JS.
const modes = {
  observe: {
    title: 'Let the world correct the model.',
    text: 'Real depth, touch, and joint history update the recurrent belief. This observation path is part of the verified original-policy replay.'
  },
  imagine: {
    title: 'Let a short future inform the action.',
    text: 'The actor and RSSM prior alternate for four predicted steps, without future sensor evidence. Reward, survival, and terminal value guide the actor update. This training path is under development.'
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
