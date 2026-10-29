<script setup>
import { ref } from 'vue'

// ref 让数字变成响应式状态，初始值为 0。
const count = ref(0)

function increment() {
  // 在 JavaScript 中，通过 .value 读写 ref。
  count.value += 1
}

function decrement() {
  // 按钮禁用之外，也在函数中守住不小于 0 的边界。
  if (count.value > 0) {
    count.value -= 1
  }
}

function reset() {
  count.value = 0
}
</script>

<template>
  <div class="counter-demo">
    <p class="counter-label">当前计数</p>
    <!-- 模板会自动解包 ref，这里直接写 count。 -->
    <output class="counter-value" aria-live="polite" aria-label="当前计数">{{ count }}</output>
    <div class="counter-actions">
      <button class="counter-button" type="button" :disabled="count === 0" @click="decrement">
        − 1
      </button>
      <button class="counter-button primary" type="button" @click="increment">+ 1</button>
    </div>
    <button class="reset-button" type="button" @click="reset">重置为 0</button>
    <div class="state-inspector" aria-label="实时状态">
      <div><span>count.value</span><code>{{ count }}</code></div>
      <div><span>减法按钮 disabled</span><code>{{ count === 0 }}</code></div>
    </div>
  </div>
</template>
