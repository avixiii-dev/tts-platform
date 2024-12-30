import { useState } from 'react'
import { 
  Container, 
  Box, 
  TextField, 
  Button, 
  Slider, 
  Typography,
  Paper
} from '@mui/material'
import axios from 'axios'

function App() {
  const [text, setText] = useState('')
  const [speed, setSpeed] = useState(1)
  const [pitch, setPitch] = useState(1)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleSubmit = async () => {
    setIsProcessing(true)
    try {
      const response = await axios.post('http://localhost:8000/tts', {
        text,
        speed,
        pitch,
        emotion: 'neutral'
      }, {
        responseType: 'blob'
      })

      const url = window.URL.createObjectURL(response.data)
      const audio = new Audio(url)
      audio.play()
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Text-to-Speech Platform
        </Typography>
        
        <Paper sx={{ p: 3, mb: 3 }}>
          <TextField
            fullWidth
            multiline
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            label="Enter text to convert to speech"
            variant="outlined"
            margin="normal"
          />

          <Box sx={{ mt: 3 }}>
            <Typography gutterBottom>Speed</Typography>
            <Slider
              value={speed}
              onChange={(_, newValue) => setSpeed(newValue as number)}
              min={0.5}
              max={2}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Box>

          <Box sx={{ mt: 3 }}>
            <Typography gutterBottom>Pitch</Typography>
            <Slider
              value={pitch}
              onChange={(_, newValue) => setPitch(newValue as number)}
              min={0.5}
              max={2}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Box>

          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={!text || isProcessing}
            sx={{ mt: 3 }}
          >
            {isProcessing ? 'Converting...' : 'Convert to Speech'}
          </Button>
        </Paper>
      </Box>
    </Container>
  )
}

export default App
