"""
Example implementation plan for demonstration purposes.
"""

from datetime import datetime
from src.models import (
    ChangeType,
    CodeChange,
    FeatureScope,
    ImplementationPlan
)

def create_example_implementation_plan():
    """Create an example implementation plan for demonstration."""
    
    # Create feature scope
    scope = FeatureScope(
        affected_files=[
            "src/App.js",
            "src/components/Header.js",
            "src/styles/theme.js",
            "src/styles/globalStyles.js",
            "src/context/ThemeContext.js"
        ],
        new_files=[
            "src/hooks/useDarkMode.js",
            "src/components/ThemeToggle.js"
        ],
        dependencies_needed=[
            "styled-components@5.3.6"
        ],
        config_changes=[
            "Update package.json with new dependency"
        ]
    )
    
    # Create code changes
    changes = [
        CodeChange(
            file_path="src/hooks/useDarkMode.js",
            change_type=ChangeType.ADD,
            description="Create a custom hook to manage dark mode state and detection",
            code_snippet="""import { useState, useEffect } from 'react';

const useDarkMode = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage first
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode !== null) {
      return savedMode === 'true';
    }
    
    // Otherwise check system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Update when system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      // Only update if no manual override in localStorage
      if (localStorage.getItem('darkMode') === null) {
        setIsDarkMode(mediaQuery.matches);
      }
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Toggle function
  const toggle = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    localStorage.setItem('darkMode', String(newMode));
  };

  return [isDarkMode, toggle];
};

export default useDarkMode;""",
            line_range=None
        ),
        CodeChange(
            file_path="src/context/ThemeContext.js",
            change_type=ChangeType.ADD,
            description="Create a context to provide theme information throughout the app",
            code_snippet="""import React, { createContext, useContext } from 'react';
import useDarkMode from '../hooks/useDarkMode';
import { lightTheme, darkTheme } from '../styles/theme';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
  const [isDarkMode, toggleDarkMode] = useDarkMode();
  const theme = isDarkMode ? darkTheme : lightTheme;
  
  return (
    <ThemeContext.Provider value={{ theme, isDarkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);""",
            line_range=None
        ),
        CodeChange(
            file_path="src/styles/theme.js",
            change_type=ChangeType.MODIFY,
            description="Define light and dark themes",
            code_snippet="""export const lightTheme = {
  primary: '#1976d2',
  secondary: '#dc004e',
  background: '#ffffff',
  surface: '#f5f5f5',
  text: '#333333',
  error: '#f44336',
  // Add more colors as needed
};

export const darkTheme = {
  primary: '#90caf9',
  secondary: '#f48fb1',
  background: '#121212',
  surface: '#1e1e1e',
  text: '#f5f5f5',
  error: '#ef9a9a',
  // Add more colors as needed
};

// ... existing code ...""",
            line_range="1-20"
        ),
        CodeChange(
            file_path="src/App.js",
            change_type=ChangeType.MODIFY,
            description="Wrap the app with the ThemeProvider and apply theme styles",
            code_snippet="""import React from 'react';
import { ThemeProvider as StyledThemeProvider } from 'styled-components';
import { ThemeProvider } from './context/ThemeContext';
import GlobalStyle from './styles/globalStyles';
import { useTheme } from './context/ThemeContext';

// ... existing imports ...

const ThemedApp = () => {
  const { theme } = useTheme();
  
  return (
    <StyledThemeProvider theme={theme}>
      <GlobalStyle />
      {/* Rest of your app components */}
      <Router>
        <Header />
        <main>
          {/* Routes and other components */}
        </main>
        <Footer />
      </Router>
    </StyledThemeProvider>
  );
};

const App = () => {
  return (
    <ThemeProvider>
      <ThemedApp />
    </ThemeProvider>
  );
};

export default App;""",
            line_range="1-35"
        )
    ]
    
    # Create implementation plan
    implementation_plan = ImplementationPlan(
        feature_name="Dark Mode Support",
        description="Add dark mode support to the UI components with automatic detection of system preferences and a manual toggle.",
        scope=scope,
        changes=changes,
        estimated_complexity="Medium",
        dependencies=[
            "The useDarkMode hook must be implemented before the ThemeContext",
            "The ThemeContext must be implemented before the ThemeToggle component",
            "The theme files must be updated before the app can be wrapped with the theme provider"
        ],
        generated_at=datetime.now()
    )
    
    return implementation_plan
