<script setup lang="ts">
import { computed, ref } from 'vue'
import axios from 'axios'

// --- State Management ---
const file = ref<File | null>(null)
const fileId = ref<string | null>(null) // Menyimpan ID unik file setelah dianalisis
const totalChunks = ref(0)
const selectedChunk = ref<number | null>(null)
const targetLanguage = ref<string | null>('Indonesian') // Default language
const translationResult = ref({ output: '', original: '' })
const loading = ref(false)

// --- Error Handling State ---
const showError = ref(false)
const errorMessage = ref('')

// --- Static Data ---
// Sesuaikan dengan Enum di backend FastAPI
const supportedLanguages = ['English', 'Indonesian', 'Malay', 'Japanese', 'Korean']

// --- Computed Properties ---
const chunkOptions = computed(() => {
  // Membuat array angka dari 1 sampai totalChunks
  return Array.from({ length: totalChunks.value }, (_, i) => i + 1)
})

const API_BASE_URL = 'http://localhost:8000'

// --- Functions ---
const handleError = (message: string) => {
  errorMessage.value = message
  showError.value = true
}

const resetState = () => {
  translationResult.value = { output: '', original: '' }
  selectedChunk.value = null
  totalChunks.value = 0
  fileId.value = null
}

const onFileChange = async () => {
  resetState()
  if (!file.value)
    return

  const formData = new FormData()

  formData.append('file', file.value)

  try {
    loading.value = true

    const { data } = await axios.post(`${API_BASE_URL}/total-chunk`, formData)

    totalChunks.value = data.total
    fileId.value = data.file_id // Simpan file_id yang diterima dari backend
    if (totalChunks.value > 0)
      selectedChunk.value = 1 // Otomatis pilih chunk pertama
  }
  catch (e) {
    console.error(e)
    handleError('Failed to analyze the EPUB file. Please try again.')
  }
  finally {
    loading.value = false
  }
}

const submitTranslation = async () => {
  if (!fileId.value || selectedChunk.value === null || !targetLanguage.value)
    return

  const formData = new FormData()

  // Kirim file_id, bukan file-nya lagi
  formData.append('file_id', fileId.value)
  formData.append('chunk', selectedChunk.value.toString())
  formData.append('target_language', targetLanguage.value)

  try {
    loading.value = true

    const { data } = await axios.post(`${API_BASE_URL}/process-chunk`, formData)

    translationResult.value = {
      output: data.output,
      original: data.original,
    }
  }
  catch (e) {
    console.error(e)
    handleError('Translation failed. Please check the server or try again.')
  }
  finally {
    loading.value = false
  }
}
</script>

<template>
  <VContainer>
    <VCard
      class="pa-6"
      elevation="4"
    >
      <VCardTitle class="text-h5 font-weight-bold">
        EPUB Translator
      </VCardTitle>
      <VCardSubtitle>Upload an EPUB file and translate it sentence by sentence.</VCardSubtitle>
      <VDivider class="my-4" />

      <!-- Form Utama -->
      <VForm @submit.prevent="submitTranslation">
        <!-- 1. Upload File -->
        <VFileInput
          v-model="file"
          label="Upload EPUB File"
          accept=".epub"
          prepend-icon="mdi-book-open-page-variant"
          variant="outlined"
          :disabled="loading"
          :clearable="!loading"
          @change="onFileChange"
        />

        <!-- 2. Pilih Bahasa & Chunk (muncul setelah file dianalisis) -->
        <VExpandTransition>
          <div v-if="totalChunks > 0">
            <!-- Pilih Bahasa -->
            <VSelect
              v-model="targetLanguage"
              :items="supportedLanguages"
              label="Select Target Language"
              class="mt-4"
              variant="outlined"
              prepend-inner-icon="mdi-translate"
              :disabled="loading"
            />

            <!-- Pilih Chunk -->
            <VSelect
              v-model="selectedChunk"
              :items="chunkOptions"
              label="Select Chunk to Translate"
              class="mt-4"
              variant="outlined"
              prepend-inner-icon="mdi-format-list-numbered"
              :disabled="loading"
            />
          </div>
        </VExpandTransition>

        <!-- Tombol Terjemahkan -->
        <VBtn
          color="primary"
          class="mt-6"
          :loading="loading"
          :disabled="!file || selectedChunk === null || !targetLanguage"
          type="submit"
          block
          size="large"
        >
          <VIcon left>
            mdi-text-box-search-outline
          </VIcon>
          Translate Chunk
        </VBtn>
      </VForm>

      <!-- Hasil Terjemahan -->
      <div
        v-if="translationResult.output"
        class="mt-6"
      >
        <VDivider class="mb-4" />
        <h3 class="text-h6 mb-2">
          Result
        </h3>

        <VCard
          variant="tonal"
          class="pa-4"
        >
          <p class="font-weight-bold">
            Original (Arabic):
          </p>
          <p class="mb-3 text-medium-emphasis">
            {{ translationResult.original }}
          </p>

          <p class="font-weight-bold">
            Translation ({{ targetLanguage }}):
          </p>
          <p class="text-body-1">
            {{ translationResult.output }}
          </p>
        </VCard>
      </div>

      <!-- Notifikasi Error -->
      <VSnackbar
        v-model="showError"
        color="error"
        timeout="5000"
      >
        {{ errorMessage }}
        <template #actions>
          <VBtn
            color="white"
            variant="text"
            @click="showError = false"
          >
            Close
          </VBtn>
        </template>
      </VSnackbar>
    </VCard>
  </VContainer>
</template>

<style scoped>
.v-card {
  border-radius: 12px;
}
</style>
