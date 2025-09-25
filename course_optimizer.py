if solution['status'] == 'Optimal':
        # Course Assignment Matrix - All Terms Combined (First Priority)
        st.subheader("📊 Course Assignment Matrix - All Terms Combined")
        st.markdown("**Binary matrix showing course-professor assignments (1 = assigned, 0 = not assigned)**")
        
        # Create comprehensive assignment matrix (courses x professors)
        # Initialize with zeros to ensure clean display
        all_assignments_matrix = pd.DataFrame(index=courses, columns=professors)
        all_assignments_matrix[:] = 0  # Explicit initialization for clean 1/0 display
        
        # Fill in assignments from all terms
        for (course, term), professor in solution['assignments'].items():
            all_assignments_matrix.loc[course, professor] = 1
        
        # Convert to integers for clean display
        all_assignments_matrix = all_assignments_matrix.astype(int)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_assignments = all_assignments_matrix.sum().sum()
            st.metric("Total Assignments", total_assignments)
        with col2:
            courses_covered = (all_assignments_matrix.sum(axis=1) > 0).sum()
            st.metric("Courses Covered", f"{courses_covered}/{len(courses)}")
        with col3:
            professors_teaching = (all_assignments_matrix.sum(axis=0) > 0).sum()
            st.metric("Professors Teaching", f"{professors_teaching}/{len(professors)}")
        
        # Display as large matrix optimized for 60 courses × 24 staff
        st.dataframe(
            all_assignments_matrix.style.applymap(
                lambda x: 'background-color: #90EE90' if x == 1 else 'background-color: #f0f0f0'
            ),
            height=min(800, max(400, len(courses) * 25)),  # Dynamic height
            use_container_width=True
        )
        
        # Individual Term Assignment Matrices
        st.subheader("📅 Individual Term Assignment Matrices")
        st.markdown("**View assignments by specific term**")
        
        # Create tabs for each term
        term_tabs = st.tabs([f"📌 {term}" for term in terms])
        
        for idx, term in enumerate(terms):
            with term_tabs[idx]:
                st.markdown(f"### Assignment Matrix for {term}")
                st.markdown("**1 = Course assigned to professor in this term, 0 = Not assigned**")
                
                # Create term-specific matrix
                term_matrix = pd.DataFrame(index=courses, columns=professors)
                term_matrix[:] = 0  # Explicit initialization
                
                # Fill in assignments for this specific term
                for (course, assigned_term), professor in solution['assignments'].items():
                    if assigned_term == term:
                        term_matrix.loc[course, professor] = 1
                
                # Convert to integers for clean display
                term_matrix = term_matrix.astype(int)
                
                # Term summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    courses_in_term = term_matrix.sum().sum()
                    st.metric(f"Courses in {term}", courses_in_term)
                with col2:
                    active_professors = (term_matrix.sum(axis=0) > 0).sum()
                    st.metric(f"Active Professors", active_professors)
                with col3:
                    total_streams_term = sum([course_streams.get((course, term), 0) 
                                            for course in courses 
                                            if (course, term) in solution['assignments']])
                    st.metric(f"Total Streams", total_streams_term)
                with col4:
                    # Calculate average workload
                    avg_courses_per_prof = courses_in_term / active_professors if active_professors > 0 else 0
                    st.metric(f"Avg Load/Prof", f"{avg_courses_per_prof:.1f}")
                
                # Display term matrix with color coding
                st.dataframe(
                    term_matrix.style.applymap(
                        lambda x: 'background-color: #90EE90' if x == 1 else 'background-color: #f0f0f0'
                    ),
                    height=min(600, max(300, len(courses) * 20)),  # Dynamic height
                    use_container_width=True
                )
