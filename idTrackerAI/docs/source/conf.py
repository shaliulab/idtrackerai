# -*- coding: utf-8 -*-
import sys
import os
import matplotlib
if os.name == 'posix':
    matplotlib.use('TkAgg')
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../utils'))
sys.path.insert(0, os.path.abspath('../../preprocessing'))
sys.path.insert(0, os.path.abspath('../../network'))
sys.path.insert(0, os.path.abspath('../../network/crossings_detector_model'))
sys.path.insert(0, os.path.abspath('../../network/identification_model'))
sys.path.insert(0, os.path.abspath('../../plots'))
sys.path.insert(0, os.path.abspath('../../plots/old_plots'))
sys.path.insert(0, os.path.abspath('../../postprocessing'))
sys.path.insert(0, os.path.abspath('../../groundtruth_utils'))
sys.path.insert(0, os.path.abspath('../../tf_cnnvisualisation'))
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'idtrackerai'
copyright = u'2018, Champalimaud Center for the Unknown'
author = u'Mattia G. Bergomi, Francisco Romero Ferrero'
version = u'0.0'
release = u'0.0.1'
language = 'en'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = False
themedir = os.path.join(os.pardir, 'scipy-sphinx-theme', '_theme')
html_theme = 'scipy'
html_theme_path = [themedir]
html_theme_options = {
        "edit_link": False,
        "sidebar": "left",
        "scipy_org_logo": True,
        "navigation_links" : True,
        "rootlinks": [("http://www.gitlab.com/polaviejalab/idtracker", "GitLab repo")]
    }
html_sidebars = {
            '**': [
                    'localtoc.html',
                    'sourcelink.html',
                    'searchbox.html'
                    ]
                }
html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

htmlhelp_basename = 'idtrackerai'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}
latex_documents = [
    (master_doc, 'idtrackerai.tex', u'idtrackerai Documentation',
     u'Mattia G. Bergomi, Francisco Romero Ferrero', 'manual'),
]
man_pages = [
    (master_doc, 'idtrackerai', u'idtrackerai Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'idtrackerai', u'idtrackerai Documentation',
     author, 'idtrackerai', 'One line description of project.',
     'Miscellaneous'),
]
