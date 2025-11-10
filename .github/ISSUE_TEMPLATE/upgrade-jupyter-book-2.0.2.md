# Upgrade to jupyter-book 2.0.2

## Current Status
- **Current version**:**: `jupyter-book==1.0.4.post1`
- **Target version**: `jupyter-book==2.0.2`
- **Files to update**:
  - `book/requirements.txt`
  - `pyproject.toml` (dev-dependencies)

## Breaking Changes in Jupyter Book 2.0

Jupyter Book 2.0 introduces significant architectural changes:

1. **MyST Document Engine**: Transition from Sphinx-based to MyST-based document engine
2. **Configuration Migration**: 
   - `_config.yml` and `_toc.yml` → `myst.yml` format
   - Configuration syntax has changed
3. **Extension Compatibility**: Some Sphinx extensions may need updates or replacements

## Tasks

### 1. Update Dependencies
- [ ] Update `book/requirements.txt`: `jupyter-book==1.0.4.post1` → `jupyter-book==2.0.2`
- [ ] Update `pyproject.toml` dev-dependencies: `jupyter-book>=1.0.4.post1` → `jupyter-book>=2.0.2`

### 2. Configuration Migration
- [ ] Review `book/en/_config.yml` and migrate to new format
- [ ] Review `book/ko/_config.yml` and migrate to new format
- [ ] Review `book/en/_toc.yml` and migrate to new format
- [ ] Review `book/ko/_toc.yml` and migrate to new format
- [ ] Check if `myst.yml` files need to be created

### 3. Extension Compatibility Check
Verify compatibility of current extensions:
- [ ] `sphinx-inline-tabs`
- [ ] `sphinx-examples`
- [ ] `sphinx-proof`
- [ ] `sphinx-hoverxref`
- [ ] `sphinxcontrib-youtube`
- [ ] `sphinxcontrib-video`
- [ ] `sphinx-thebe`
- [ ] `sphinxcontrib-mermaid`
- [ ] `sphinx-carousel`
- [ ] `sphinxcontrib-lastupdate`

### 4. Build Scripts
- [ ] Review `book/_scripts/build.sh` for any needed changes
- [ ] Test build process with new version

### 5. Testing
- [ ] Build English documentation: `jupyter-book build book/en`
- [ ] Build Korean documentation: `jupyter-book build book/ko`
- [ ] Verify all content renders correctly
- [ ] Check interactive elements (videos, mermaid diagrams, etc.)
- [ ] Test language switcher functionality
- [ ] Verify search functionality

### 6. Documentation
- [ ] Update any references to jupyter-book version in documentation
- [ ] Update build instructions if needed

## Resources
- [Jupyter Book 2.0 Upgrade Guide](https://jupyterbook.org/latest/)
- [MyST Documentation](https://myst.tools/docs/)
- [Migration Tools](https://jupyterbook.org/latest/start/migrate.html)

## Notes
- This is a major version upgrade with breaking changes
- Both English and Korean documentation need to be tested
- Multi-language setup may require special attention during migration

