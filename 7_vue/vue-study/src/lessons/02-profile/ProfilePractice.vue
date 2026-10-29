<script setup>
import { computed, reactive } from 'vue'

// 先读 ProfileDemo.vue。computed 导入已准备好，供 TODO 2、3 使用。
const profile = reactive({
  name: 'Vue 学习者',
  role: '前端入门',
  bio: '每天写一点 Vue。',
})

// TODO 2：将这两个占位值替换为 computed。
// displayName：昵称去除首尾空白，空值显示「未命名同学」。
// bioLength：根据 profile.bio 自动计算字符数。
const displayName = '待实现的昵称'
const bioLength = 0

// TODO 3：把以下三个占位值替换为 computed。
// nameError：去除首尾空白后，昵称为空或超过 12 个字符时给出提示，否则返回空字符串。
// bioError：自我介绍超过 80 个字符时给出提示，否则返回空字符串。
// isValid：只有两项错误都为空时才为 true。在脚本中读 computed 需要 .value。
const nameError = ''
const bioError = ''
const isValid = true

function reset() {
  profile.name = 'Vue 学习者'
  profile.role = '前端入门'
  profile.bio = '每天写一点 Vue。'
}
</script>

<template>
  <div class="profile-editor">
    <form class="profile-form" @submit.prevent>
      <label for="practice-name">昵称</label>
      <!-- TODO 1：将昵称、学习阶段、自我介绍的 :value 改为 v-model。 -->
      <!-- :value 只把数据放进输入框；你还需要让输入更新 profile。 -->
      <input id="practice-name" :value="profile.name" type="text" autocomplete="off" :aria-invalid="Boolean(nameError)" aria-describedby="practice-name-help" />
      <p id="practice-name-help" class="field-help" :class="{ 'field-error': nameError }">{{ nameError || '昵称去除首尾空白后需为 1—12 个字符。' }}</p>

      <label for="practice-role">学习阶段</label>
      <select id="practice-role" :value="profile.role">
        <option>前端入门</option>
        <option>正在练习</option>
        <option>能写小项目</option>
      </select>

      <label for="practice-bio">自我介绍</label>
      <textarea id="practice-bio" :value="profile.bio" rows="3" :aria-invalid="Boolean(bioError)" aria-describedby="practice-bio-help"></textarea>
      <p id="practice-bio-help" class="field-help" :class="{ 'field-error': bioError }">{{ bioError || '自我介绍最多 80 个字符，可以留空。' }} 已输入 {{ bioLength }} / 80</p>
      <button class="secondary-button" type="button" @click="reset">重置资料</button>
    </form>

    <aside class="profile-preview" aria-label="练习资料预览">
      <span class="small-label">我的资料预览</span>
      <h3>{{ displayName }}</h3>
      <span class="profile-role">{{ profile.role }}</span>
      <p class="profile-bio">{{ profile.bio || '还没有写自我介绍。' }}</p>
      <p class="validation-status" :class="{ invalid: !isValid }" role="status">{{ isValid ? '资料可用' : '请先修正输入' }}</p>
      <p class="preview-note">状态提示也属于练习任务，完成 TODO 3 后才会正确校验。</p>
    </aside>
  </div>
</template>
