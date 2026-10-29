<script setup>
import { ref } from 'vue'

const count = ref(0)
const step = ref(2)
const maxCount = 10

function increment() {
  // 先计算新值，再限制上界；9 + 2 会停在 10。
  count.value = Math.min(maxCount, count.value + step.value)
}

function decrement() {
  // 只禁用按钮还不够：1 - 2 也需要守住下界。
  count.value = Math.max(0, count.value - step.value)
}

function reset() {
  count.value = 0
  step.value = 2
}
</script>

<template>
  <div class="counter-demo">
    <p class="counter-label">完成后的计数器</p>
    <output class="counter-value" aria-live="polite" aria-label="答案计数">{{ count }}</output>
    <div class="step-picker" role="group" aria-label="选择步长">
      <span>步长</span>
      <button type="button" :aria-pressed="step === 1" @click="step = 1">1</button>
      <button type="button" :aria-pressed="step === 2" @click="step = 2">2</button>
      <button type="button" :aria-pressed="step === 3" @click="step = 3">3</button>
    </div>
    <div class="counter-actions">
      <button class="counter-button" type="button" :disabled="count === 0" @click="decrement">− {{ step }}</button>
      <button class="counter-button primary" type="button" :disabled="count >= maxCount" @click="increment">+ {{ step }}</button>
    </div>
    <button class="reset-button" type="button" @click="reset">重置计数和步长</button>
    <div class="state-inspector" aria-label="答案实时状态">
      <div><span>count.value</span><code>{{ count }}</code></div>
      <div><span>step.value</span><code>{{ step }}</code></div>
      <div><span>上限 maxCount</span><code>{{ maxCount }}</code></div>
    </div>
  </div>
</template>
